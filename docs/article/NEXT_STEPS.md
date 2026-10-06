# Research readiness and next steps (2026-10-05)

## What this repair establishes

This is a research prototype, not a validated independent mobility aid. The
repair restores the saved project and makes several previously untested
assumptions explicit. Historical results remain historical; no new detection
accuracy or human navigation results are claimed.

Verification on 2026-10-05: 205 tests passed without skips; Expo App.js
compiled with Babel; a real YOLOv8n CPU API run processed three frames and
returned Russian audio. A separate Kazakh synthesis produced a valid
0.8-second WAV. Routing was rebuilt from the saved OSM snapshot. Dependency
consistency passed. These are engineering checks, not field evaluation.

The Windows eSpeak data-path failure under the Cyrillic project directory
was reproduced and resolved with a versioned ASCII cache under LOCALAPPDATA.
YOLO predictor/tracker warmup now happens before readiness is announced.

The 82 missing tracked files were recovered from commit 918abfe without
overwriting existing files. The recovery manifest is recovery_2026-10-05.json.

## Priority 1: reproducible engineering

- Run the complete unit suite and the real-model API smoke test.
- Preserve the dependency snapshot, model identity, configuration, device,
  camera calibration and input manifest with each experiment.
- Validate the actual iPhone connection, Russian/Kazakh audio, disconnect
  behavior, camera failure and latency before recording participant data.
- Keep disabled, failed and operational depth modes distinct. A CPU smoke
  without depth is not a validation of depth-based hazards.
- Measure camera capture to audible output on the phone; server processing
  timings exclude image acquisition, transfer and playback.

## Priority 2: independent data

The original kz_hazards export has 3,258 images. It mixes YOLO rectangles
and polygon annotations. The repaired preparation script converts polygons
to enclosing rectangles, validates geometry, and saves a new version without
changing original images or annotations.

Known video frames, Roboflow filename variants and byte-identical images
are kept in one split. The grouped candidate contains train/valid/test image
counts of 2,117/96/1,045. Eighteen inferred groups crossed original splits.
Grouping is based on names and hashes; it does not prove complete scene
independence. Numbered stairs images still require origin/scene review.

All 2,301 sidewalk-obstacle images come from two filename-identified video
sessions. One is held for testing and one for training. The validation set
therefore has no obstacle instances. Collect at least one additional
independent session to make three-way session separation possible; a robust
study needs substantially more sessions, locations and conditions.

Do not treat that minimum as a statistically sufficient sample size. Choose
the final sample size after a pilot and report uncertainty by independent
walk/session, not by treating adjacent video frames as independent samples.

Retrieve exact source dataset URLs, versions, licences and attribution before
publishing or redistributing data. The name kz_hazards is not evidence of
Kazakhstan provenance. Collect a new external holdout because historical
test images have already influenced model development.

Do not report the old mAP values as results for the new split. Training is
blocked by default while the manifest is not benchmark-ready. Exploratory
runs require --exploratory and are labelled in their output. Their weights
stay with the run instead of replacing the live model.

## Priority 3: the paper's actual contribution

Working title: Risk-Aware Fusion of Visual Perception and Pedestrian Routing
for Assistive Navigation.

Hypothesis: combining local risk, route alignment and stable guidance may
reduce hazardous recommendations and route deviation under a limited speech
budget. This is a testable hypothesis, not an established superiority claim.

Evaluate the full configuration, route-only, vision-only, no-depth,
no-smoothing and no-speech-scheduling variants on the same recordings.
Check that each ablation actually disables the claimed component.

Report detector precision/recall and mAP by class, event-level hazard recall,
false warnings per minute, distance error against measured ground truth,
end-to-end latency p50/p95, speech occupancy and contradictory guidance.
Route completion and user workload need actual supervised user trials;
offline replay cannot establish them.

Record video and pose timestamps together. Replay now uses recorded frame
timestamps instead of manufacturing them from container FPS. Old replay
metrics that depended on timing must be regenerated.

Before participant studies, obtain institutional ethics review and informed
consent as applicable, plan accessible instructions and supervised trials.
Blindfolded sighted participants are not a substitute for evaluation with
the intended users. First conduct controlled, accompanied engineering tests.

## What needs the author's input

1. Target journal and author/affiliation information.
2. Source dataset provenance/licences and additional independent recordings.
3. A physical iPhone test with headphones and confirmed camera mounting.
4. Approved user-study protocol, participants and actual field observations.

The program can prepare and analyse these data; it cannot manufacture them.
