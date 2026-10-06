/**
 * blind-nav — мобильный клиент.
 *
 * РОЛЬ УСТРОЙСТВА
 * Телефон — датчики и звук, а не вычислитель. Камера, GPS и компас
 * уходят на ноутбук, обратно приходит готовая речь. Причина в платформе:
 * нативную iOS-сборку без Mac не собрать, а в Expo Go нет нативных
 * модулей для YOLO на устройстве.
 *
 * ПОЧЕМУ ЭКРАН ПОКАЗЫВАЕТ КАМЕРУ
 * Пользователю экран не нужен — он слушает. Но на экран смотрят трое
 * других: разработчик при отладке, сопровождающий при полевом выходе
 * и сам исследователь при разборе прогулки. Скрытая камера лишает всех
 * троих возможности понять, ЧТО именно видит система и почему молчит.
 *
 * Поэтому: превью камеры во весь верх, поверх него — последняя реплика
 * и состояние. Ниже — диагностика, по которой видно, на чём система
 * остановилась.
 *
 * ГЛАВНОЕ ПРАВИЛО ЭКРАНА
 * Ни одно состояние не выглядит как работа, если работы нет. Пустой
 * экран, надпись по умолчанию или молчание при отсутствии связи —
 * это ложь пользователю. Каждое состояние названо явно.
 */

import React, { useCallback, useEffect, useRef, useState } from 'react';
import {
  Linking, Pressable, SafeAreaView, ScrollView, StyleSheet, Text, TextInput,
  Vibration, View,
} from 'react-native';
import { StatusBar } from 'expo-status-bar';
import Constants from 'expo-constants';
import { CameraView, useCameraPermissions } from 'expo-camera';
import * as Location from 'expo-location';
import * as Haptics from 'expo-haptics';
import { Audio } from 'expo-av';

const PORT = 8765;
const JPEG_QUALITY = 0.45;

function detectLaptopHost() {
  const candidates = [
    Constants.expoConfig?.hostUri,
    Constants.expoGoConfig?.debuggerHost,
    Constants.manifest2?.extra?.expoGo?.debuggerHost,
  ];
  for (const c of candidates) {
    if (typeof c === 'string' && c.length) {
      const host = c.split(':')[0];
      if (host) return host;
    }
  }
  return '172.20.10.2';
}

export default function App() {
  const [host, setHost] = useState(detectLaptopHost());
  const [permission, requestPermission] = useCameraPermissions();
  const [running, setRunning] = useState(false);
  const [conn, setConn] = useState('нет');        // нет | подключаюсь | есть
  const [gpsReady, setGpsReady] = useState(false);
  const [gpsAcc, setGpsAcc] = useState(null);
  const [status, setStatus] = useState({});
  const [utterance, setUtterance] = useState(null);
  const [latencyMs, setLatencyMs] = useState(null);
  const [captureMs, setCaptureMs] = useState(null);
  const [sent, setSent] = useState(0);
  const [targetQuery, setTargetQuery] = useState('');
  const [log, setLog] = useState([]);

  const cameraRef = useRef(null);
  const socketRef = useRef(null);
  const soundRef = useRef(null);
  const seqRef = useRef(0);
  const sentAtRef = useRef(new Map());
  const inFlightRef = useRef(false);
  const poseRef = useRef({ lat: null, lon: null, heading: null, accuracy: 30 });
  const runningRef = useRef(false);
  const lastSentAtRef = useRef(0);
  const sendFrameRef = useRef(null);
  const connectRef = useRef(null);

  const addLog = useCallback((line) => {
    setLog((prev) => [`${new Date().toLocaleTimeString()}  ${line}`, ...prev].slice(0, 60));
  }, []);

  // --- разрешения спрашиваем СРАЗУ, а не по нажатию -----------------------
  //
  // iOS показывает диалог только ОДИН раз. Если пользователь когда-то
  // отказал, повторный запрос молча возвращает отказ, не показывая
  // ничего — и человек видит приложение, которое не работает без причины.
  // Поэтому запрашиваем при запуске и явно показываем результат.
  useEffect(() => {
    (async () => {
      if (!permission?.granted) {
        const r = await requestPermission();
        addLog(r?.granted ? 'камера разрешена'
                          : 'КАМЕРА ЗАПРЕЩЕНА — Настройки → Expo Go → Камера');
      } else {
        addLog('камера разрешена');
      }
    })();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // --- датчики -----------------------------------------------------------
  useEffect(() => {
    let posSub, headSub;
    (async () => {
      const { status: st } = await Location.requestForegroundPermissionsAsync();
      if (st !== 'granted') {
        addLog('ГЕОПОЗИЦИЯ ЗАПРЕЩЕНА — Настройки → Expo Go → Геопозиция');
        return;
      }
      posSub = await Location.watchPositionAsync(
        { accuracy: Location.Accuracy.BestForNavigation, distanceInterval: 1, timeInterval: 1000 },
        (loc) => {
          const first = poseRef.current.lat == null;
          poseRef.current.lat = loc.coords.latitude;
          poseRef.current.lon = loc.coords.longitude;
          poseRef.current.accuracy = loc.coords.accuracy ?? 30;
          setGpsAcc(loc.coords.accuracy ?? null);
          setGpsReady(true);
          if (first) addLog(`GPS получен, точность ${Math.round(loc.coords.accuracy ?? 0)} м`);
        },
      );
      headSub = await Location.watchHeadingAsync((h) => {
        poseRef.current.heading = h.trueHeading >= 0 ? h.trueHeading : h.magHeading;
      });
    })();
    return () => { posSub?.remove?.(); headSub?.remove?.(); };
  }, [addLog]);

  // Сторож: если кадры перестали уходить, разбудить съёмку.
  // Молчаливая остановка недопустима — человек продолжает идти,
  // считая, что его прикрывают.
  useEffect(() => {
    const id = setInterval(() => {
      if (!runningRef.current) return;
      if (Date.now() - lastSentAtRef.current > 3000) {
        addLog('съёмка встала — перезапускаю');
        inFlightRef.current = false;
        lastSentAtRef.current = Date.now();
        const sock = socketRef.current;
        if (!sock || sock.readyState !== 1) connectRef.current?.();
        else sendFrameRef.current?.();
      }
    }, 1500);
    return () => clearInterval(id);
  }, [addLog]);

  useEffect(() => {
    Audio.setAudioModeAsync({
      playsInSilentModeIOS: true,      // аид обязан звучать при беззвучном режиме
      staysActiveInBackground: false,
      shouldDuckAndroid: true,
    }).catch(() => {});
  }, []);

  // --- воспроизведение ---------------------------------------------------
  const play = useCallback(async (base64, interrupt) => {
    try {
      if (interrupt && soundRef.current) {
        // Дослушивание предыдущей фразы стоит полутора секунд —
        // двух шагов человека. Экстренное сообщение вытесняет её.
        await soundRef.current.stopAsync().catch(() => {});
        await soundRef.current.unloadAsync().catch(() => {});
        soundRef.current = null;
      }
      if (soundRef.current) return;    // обычная реплика не перебивает
      const { sound } = await Audio.Sound.createAsync(
        { uri: `data:audio/wav;base64,${base64}` }, { shouldPlay: true },
      );
      soundRef.current = sound;
      sound.setOnPlaybackStatusUpdate((s) => {
        if (s.didJustFinish) { sound.unloadAsync().catch(() => {}); soundRef.current = null; }
      });
    } catch (e) {
      addLog(`звук: ${String(e).slice(0, 60)}`);
    }
  }, [addLog]);

  const haptic = useCallback((pattern) => {
    if (pattern === 'long') Vibration.vibrate(600);
    else if (pattern === 'short-double') Vibration.vibrate([0, 80, 90, 80]);
    else Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Medium).catch(() => {});
  }, []);

  // --- отправка кадра ----------------------------------------------------
  const sendFrame = useCallback(async () => {
    if (!runningRef.current || inFlightRef.current) return;
    const sock = socketRef.current;
    if (!sock || sock.readyState !== 1) return;
    if (!cameraRef.current) { addLog('камера ещё не готова'); return; }

    // Координаты НЕ обязательны. В помещении спутников нет, и требовать
    // их значит не работать дома вообще — а дом и есть основной сценарий.
    // Решения о безопасности принимаются по кадру и курсу; координата
    // нужна только маршруту, которого в домашнем режиме нет.
    const { lat, lon } = poseRef.current;

    inFlightRef.current = true;
    try {
      // takePictureAsync делает полноценный снимок, а не кадр потока,
      // и в Expo Go иногда не возвращается вовсе. Без таймаута это
      // останавливает съёмку навсегда: inFlight остаётся поднятым,
      // и следующий кадр не запрашивается никогда.
      const t0 = Date.now();
      const photo = await Promise.race([
        cameraRef.current.takePictureAsync({
          base64: true, quality: JPEG_QUALITY, skipProcessing: true, shutterSound: false,
        }),
        new Promise((_, rej) => setTimeout(() => rej(new Error('снимок завис')), 4000)),
      ]);
      const capMs = Date.now() - t0;
      setCaptureMs(capMs);
      const seq = seqRef.current++;
      if (seq === 0) addLog(`первый снимок за ${capMs} мс`);
      sentAtRef.current.set(seq, Date.now());
      sock.send(JSON.stringify({
        seq, ts: Date.now() / 1000,
        lat: lat ?? null, lon: lon ?? null,
        heading: poseRef.current.heading,
        accuracy: poseRef.current.accuracy,
        jpeg_b64: photo.base64,
      }));
      setSent(seq + 1);
      lastSentAtRef.current = Date.now();
    } catch (e) {
      inFlightRef.current = false;
      addLog(`кадр: ${String(e).slice(0, 70)}`);
      setTimeout(() => { if (runningRef.current) sendFrame(); }, 800);
    }
  }, [addLog]);

  // --- соединение --------------------------------------------------------
  const connect = useCallback(() => {
    const url = `ws://${host}:${PORT}/ws`;
    setConn('подключаюсь');
    addLog(`подключение ${url}`);

    const sock = new WebSocket(url);
    socketRef.current = sock;

    // WebSocket не даёт таймаута сам: при отброшенных пакетах он висит
    // молча. Явный таймер превращает тишину в понятное сообщение.
    const timer = setTimeout(() => {
      if (sock.readyState === 0) {
        addLog('НЕ ОТВЕЧАЕТ. Проверь: сервер запущен? брандмауэр пускает?');
        setConn('нет');
        try { sock.close(); } catch (_) {}
      }
    }, 6000);

    sock.onopen = () => {
      clearTimeout(timer);
      setConn('есть');
      addLog('соединение установлено');
      sendFrame();
    };
    sock.onclose = (e) => {
      clearTimeout(timer);
      setConn('нет');
      addLog(`соединение закрыто${e?.code ? ` (код ${e.code})` : ''}`);
      // Обрыв в поле — норма: связь рвётся, телефон уходит в сон.
      // Молчать и не восстанавливаться нельзя.
      if (runningRef.current) setTimeout(() => { if (runningRef.current) connect(); }, 1500);
    };
    sock.onerror = (e) => {
      clearTimeout(timer);
      setConn('нет');
      addLog(`ошибка связи: ${e?.message || 'нет ответа от ноутбука'}`);
    };

    sock.onmessage = (ev) => {
      inFlightRef.current = false;
      let msg;
      try { msg = JSON.parse(ev.data); } catch { return; }

      if (msg.error) { addLog(`сервер: ${msg.error}`); }
      if (msg.ack) { addLog(`сервер подтвердил: ${msg.ack}`); return; }

      const sentAt = sentAtRef.current.get(msg.seq);
      if (sentAt) { setLatencyMs(Date.now() - sentAt); sentAtRef.current.delete(msg.seq); }

      if (msg.state) setStatus(msg.state);
      if (msg.utterance) { setUtterance(msg.utterance); addLog(`«${msg.utterance.text}»`); }
      if (msg.haptic) haptic(msg.haptic);
      if (msg.audio_b64) {
        play(msg.audio_b64, msg.interrupt || (msg.utterance?.urgency ?? 0) >= 3);
      }

      if (runningRef.current) sendFrame();
    };
  }, [host, addLog, sendFrame, play, haptic]);

  useEffect(() => { sendFrameRef.current = sendFrame; }, [sendFrame]);
  useEffect(() => { connectRef.current = connect; }, [connect]);

  const start = useCallback(async () => {
    if (!permission?.granted) {
      const r = await requestPermission();
      if (!r?.granted) {
        // Диалога уже не будет: iOS спрашивает один раз. Единственный
        // путь — настройки системы, и об этом надо сказать прямо,
        // а не оставлять человека перед неработающим приложением.
        addLog('Без камеры работать нечем. Открой Настройки и разреши доступ.');
        Linking.openSettings().catch(() => {});
        return;
      }
    }
    runningRef.current = true;
    lastSentAtRef.current = Date.now();
    setRunning(true);
    connect();
  }, [permission, requestPermission, connect, addLog]);

  const stop = useCallback(() => {
    runningRef.current = false;
    setRunning(false);
    try { socketRef.current?.close(); } catch (_) {}
    socketRef.current = null;
    setConn('нет');
  }, []);

  const sendTarget = useCallback(async (query) => {
    try {
      const r = await fetch(`http://${host}:${PORT}/target`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query }),
      });
      const j = await r.json();
      if (j.cleared) addLog('поиск отменён');
      else if (j.error) addLog(j.error);
      else addLog(`ищу: ${j.name}`);
    } catch (e) {
      addLog(`поиск: ${String(e).slice(0, 60)}`);
    }
  }, [host, addLog]);

  const confirmCrossing = useCallback(() => {
    const sock = socketRef.current;
    if (sock && sock.readyState === 1) {
      sock.send(JSON.stringify({ type: 'confirm_crossing', ts: Date.now() / 1000 }));
      Haptics.notificationAsync(Haptics.NotificationFeedbackType.Success).catch(() => {});
    }
  }, []);

  // --- что показывать крупно ---------------------------------------------
  const waiting = status.crossing_phase === 'waiting';
  const emergency = status.action === 'stop' || status.blocked;

  let headline = 'Не запущено';
  if (permission && !permission.granted) headline = 'Нет доступа к камере';
  else if (running && conn !== 'есть') headline = conn === 'подключаюсь' ? 'Подключаюсь…' : 'Нет связи';

  else if (utterance) headline = utterance.text;
  else if (running) headline = 'Слушаю обстановку';

  return (
    <SafeAreaView style={styles.root}>
      <StatusBar style="light" />

      <View style={styles.previewBox}>
        {permission?.granted ? (
          <CameraView ref={cameraRef} style={StyleSheet.absoluteFill} facing="back" />
        ) : (
          <View style={[StyleSheet.absoluteFill, styles.noCam]}>
            <Text style={styles.noCamText}>Нет доступа к камере</Text>
            <Text style={styles.noCamHint}>
              iOS спрашивает один раз. Дальше разрешить можно только
              в настройках: Настройки → Expo Go → Камера
            </Text>
            <Pressable style={styles.settingsBtn}
                       onPress={() => Linking.openSettings().catch(() => {})}
                       accessibilityRole="button"
                       accessibilityLabel="Открыть настройки">
              <Text style={styles.btnText}>Открыть настройки</Text>
            </Pressable>
          </View>
        )}

        <Pressable
          style={[styles.overlay, emergency && styles.overlayDanger, waiting && styles.overlayWait]}
          onPress={waiting ? confirmCrossing : undefined}
          accessibilityRole="button"
          accessibilityLabel={waiting ? 'Ожидание у перехода. Коснитесь, чтобы подтвердить' : headline}
        >
          <Text style={styles.headline} accessibilityLiveRegion="assertive">{headline}</Text>
          {waiting && <Text style={styles.hint}>Коснитесь, когда решите идти</Text>}
        </Pressable>

        <View style={styles.badges}>
          <Badge label="камера" value={permission?.granted ? 'да' : 'НЕТ'}
                 good={!!permission?.granted} />
          <Badge label="связь" value={conn} good={conn === 'есть'} />
          <Badge label="GPS" value={gpsReady ? `${Math.round(gpsAcc ?? 0)} м` : 'дома нет'}
                 good />
          <Badge label="кадров" value={String(sent)} good={sent > 0} />
          <Badge label="мс" value={latencyMs != null ? String(latencyMs) : '—'} good={latencyMs != null && latencyMs < 300} />
          <Badge label="снимок" value={captureMs != null ? `${captureMs}мс` : '—'}
                 good={captureMs != null && captureMs < 400} />
        </View>
      </View>

      <View style={styles.panel}>
        <View style={styles.row}>
          <Text style={styles.label}>Ноутбук</Text>
          <TextInput style={styles.input} value={host} onChangeText={setHost}
                     autoCapitalize="none" placeholder="192.168.0.2" placeholderTextColor="#666"
                     accessibilityLabel="Адрес ноутбука" />
        </View>

        <View style={styles.row}>
          <Text style={styles.label}>Найти</Text>
          <TextInput style={styles.input} value={targetQuery} onChangeText={setTargetQuery}
                     placeholder="кружка, стул, дверь" placeholderTextColor="#666"
                     autoCapitalize="none"
                     onSubmitEditing={() => sendTarget(targetQuery)}
                     accessibilityLabel="Что искать" />
          <Pressable style={styles.small} onPress={() => sendTarget(targetQuery)}
                     accessibilityRole="button" accessibilityLabel="Искать предмет">
            <Text style={styles.smallText}>Искать</Text>
          </Pressable>
          <Pressable style={styles.small}
                     onPress={() => { setTargetQuery(''); sendTarget(''); }}
                     accessibilityRole="button" accessibilityLabel="Отменить поиск">
            <Text style={styles.smallText}>Сброс</Text>
          </Pressable>
        </View>

        <Pressable style={[styles.btn, running ? styles.btnStop : styles.btnStart]}
                   onPress={running ? stop : start}
                   accessibilityRole="button"
                   accessibilityLabel={running ? 'Остановить навигацию' : 'Начать навигацию'}>
          <Text style={styles.btnText}>{running ? 'Стоп' : 'Начать'}</Text>
        </Pressable>

        <View style={styles.stats}>
          <Stat label="объектов" value={status.detections ?? '—'} />
          <Stat label="доверие" value={status.confidence != null ? status.confidence.toFixed(2) : '—'} />
          <Stat label="действие" value={status.action ?? '—'} />
          <Stat label="до манёвра" value={status.distance_to_maneuver_m != null ? `${status.distance_to_maneuver_m} м` : '—'} />
          <Stat label="ищу" value={status.target || '—'} />
          <Stat label="сошли" value={status.off_route == null ? '—' : (status.off_route ? 'да' : 'нет')} />
        </View>

        {status.flags?.length > 0 && (
          <Text style={styles.flags}>⚠ {status.flags.join(', ')}</Text>
        )}

        <ScrollView style={styles.log}>
          {log.map((l, i) => <Text key={i} style={styles.logLine}>{l}</Text>)}
        </ScrollView>
      </View>
    </SafeAreaView>
  );
}

function Badge({ label, value, good }) {
  return (
    <View style={[styles.badge, good ? styles.badgeGood : styles.badgeBad]}>
      <Text style={styles.badgeLabel}>{label}</Text>
      <Text style={styles.badgeValue}>{value}</Text>
    </View>
  );
}

function Stat({ label, value }) {
  return (
    <View style={styles.stat} accessibilityLabel={`${label}: ${value}`}>
      <Text style={styles.statLabel}>{label}</Text>
      <Text style={styles.statValue}>{String(value)}</Text>
    </View>
  );
}

/* Высокий контраст: часть пользователей имеет остаточное зрение,
   и различение крупных светлых форм на чёрном для них доступно. */
const styles = StyleSheet.create({
  root: { flex: 1, backgroundColor: '#000' },
  previewBox: { flex: 1, backgroundColor: '#111' },
  noCam: { alignItems: 'center', justifyContent: 'center' },
  noCamText: { color: '#f87171', fontSize: 20, fontWeight: '700', marginBottom: 10 },
  noCamHint: { color: '#9ca3af', fontSize: 14, textAlign: 'center',
               paddingHorizontal: 24, lineHeight: 20, marginBottom: 16 },
  settingsBtn: { backgroundColor: '#1d4ed8', paddingHorizontal: 20,
                 paddingVertical: 12, borderRadius: 10 },
  overlay: {
    position: 'absolute', left: 0, right: 0, bottom: 0,
    paddingVertical: 18, paddingHorizontal: 16,
    backgroundColor: 'rgba(0,0,0,0.68)',
  },
  overlayDanger: { backgroundColor: 'rgba(153,27,27,0.9)' },
  overlayWait: { backgroundColor: 'rgba(120,53,15,0.9)' },
  headline: { color: '#fff', fontSize: 26, fontWeight: '700', textAlign: 'center', lineHeight: 32 },
  hint: { color: '#fbbf24', fontSize: 16, marginTop: 10, textAlign: 'center' },
  badges: { position: 'absolute', top: 8, left: 8, right: 8, flexDirection: 'row', flexWrap: 'wrap' },
  badge: { paddingHorizontal: 9, paddingVertical: 4, borderRadius: 7, marginRight: 6, marginBottom: 6 },
  badgeGood: { backgroundColor: 'rgba(21,128,61,0.85)' },
  badgeBad: { backgroundColor: 'rgba(120,20,20,0.85)' },
  badgeLabel: { color: '#d1d5db', fontSize: 10 },
  badgeValue: { color: '#fff', fontSize: 14, fontWeight: '700' },
  panel: { backgroundColor: '#111', padding: 12, borderTopWidth: 1, borderTopColor: '#333' },
  row: { flexDirection: 'row', alignItems: 'center', marginBottom: 8 },
  label: { color: '#9ca3af', width: 72, fontSize: 14 },
  input: {
    flex: 1, backgroundColor: '#1f2937', color: '#fff',
    paddingHorizontal: 10, paddingVertical: 8, borderRadius: 8, fontSize: 15,
  },
  small: { marginLeft: 8, backgroundColor: '#374151', paddingHorizontal: 12, paddingVertical: 9, borderRadius: 8 },
  smallText: { color: '#fff', fontWeight: '600' },
  btn: { paddingVertical: 15, borderRadius: 10, alignItems: 'center', marginBottom: 8 },
  btnStart: { backgroundColor: '#15803d' },
  btnStop: { backgroundColor: '#b91c1c' },
  btnText: { color: '#fff', fontSize: 19, fontWeight: '700' },
  stats: { flexDirection: 'row', flexWrap: 'wrap' },
  stat: { width: '33%', paddingVertical: 3 },
  statLabel: { color: '#6b7280', fontSize: 10 },
  statValue: { color: '#e5e7eb', fontSize: 14, fontWeight: '600' },
  flags: { color: '#fbbf24', fontSize: 12, marginTop: 4 },
  log: { maxHeight: 96, marginTop: 6 },
  logLine: { color: '#6b7280', fontSize: 10 },
});
