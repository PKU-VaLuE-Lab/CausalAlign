# Third-party code

## Fast-WAM

`wam/src/fastwam/` and `wam/scripts/preprocess_action_dit_backbone.py` come from
Fast-WAM (https://github.com/yuantianyuan01/FastWAM), commit
`45d8e1458921d83f8ad6cf9ce993d371208dabd0`, MIT License, Copyright (c) 2026 The
FastWAM Authors (`FastWAM-LICENSE`). Files are unchanged from that commit except:

- `models/wan22/fastwam.py`: frozen LoRA adapters for the video and action experts,
  action-dimension masking in the training loss and in action sampling; the joint
  video-action inference methods are removed.
- `models/wan22/frozen_lora.py`: added.
- `runtime.py`, `models/wan22/helpers/state_dict_converters.py` and the package
  `__init__.py` files: reduced to the parts used here.
- `scripts/preprocess_action_dit_backbone.py`: `--redirect-common-files` option; the
  text encoder is not loaded.

The Wan model files in `models/wan22/` are distributed as part of Fast-WAM; their
upstream origin is not stated in the Fast-WAM repository.

## OpenWAM

`wam/reference_video_io.py` is copied unchanged from OpenWAM, revision
`7c5861e45cfe1339a0323f0e0b03a3316c37971c`, Apache License 2.0 (`OpenWAM-LICENSE`).

## Not included

Wan2.2 weights, LingBot-Map, VideoX-Fun, RoboCasa365, robosuite, MuJoCo and the
datasets are obtained separately under their own terms.
