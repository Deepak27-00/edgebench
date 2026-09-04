# Third-party notices

## Licence scope

EdgeBench is distributed as a whole under the GNU Affero General Public License,
version 3.0 (AGPL-3.0). The licence text is in [LICENSE](LICENSE).

The repository contains third-party model files and Python dependencies. Their
respective licence terms continue to apply. Nothing in this repository grants a
licence to third-party material beyond the permissions its rightsholders provide.

## Ultralytics YOLO model files

The following ONNX files are derived from pretrained Ultralytics YOLO models and
are not original EdgeBench models:

| File | Origin | Status |
| --- | --- | --- |
| `models/yolo11n.onnx` | Ultralytics YOLO11n pretrained weights, exported to ONNX | AGPL-3.0 terms apply |
| `models/yolo11n.int8.onnx` | Dynamically quantised derivative of `yolo11n.onnx` | AGPL-3.0 terms apply |
| `models/yolo26n.onnx` | Ultralytics YOLO26n pretrained weights, exported to ONNX | AGPL-3.0 terms apply |

Ultralytics states that its trained YOLO models are AGPL-3.0 by default and that
users of its code, models, architectures, training pipelines, or trained models
must either open-source the complete corresponding project under AGPL-3.0 or obtain
an Enterprise licence. Read the current terms at:

- <https://www.ultralytics.com/license>
- <https://github.com/ultralytics/ultralytics>

EdgeBench is not affiliated with, endorsed by, or sponsored by Ultralytics.

## Deployment obligation

If this application is deployed for users over a network, publish the complete
corresponding source for the exact deployed version, including this application,
its configuration, build files, scripts, and applicable model artefacts, under
AGPL-3.0. The web UI links to this notice and the public source repository so
users can find that source.

For proprietary, commercial, private, SaaS, or embedded deployment, obtain legal
advice and confirm the appropriate Ultralytics licence before deployment.

## Other dependencies

The application uses packages including FastAPI, Uvicorn, ONNX Runtime, NumPy,
OpenCV, psutil, Pydantic, and python-multipart. Their licence notices and source
are provided by their respective publishers. Before distributing a container or
binary, generate and retain a dependency licence inventory for the exact locked
environment you ship.
