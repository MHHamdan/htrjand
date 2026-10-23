"""HTR-JAND: line-level handwritten text recognition.

    from htrjand import HTRJANDModel
    model = HTRJANDModel.from_pretrained("MHamdan/htrjand-student", variant="general")
    print(model.transcribe("line.png").text)

Submodules are imported lazily, so `import htrjand` needs only torch and numpy;
training / evaluation modules additionally need h5py and editdistance, the T5
corrector needs transformers, the demo needs gradio.
"""
__version__ = "1.0.0"
from .release import RELEASE_VERSION as RELEASE, SOURCE_COMMIT  # noqa: E402

__all__ = ["HTRJANDModel", "Transcription", "NotALineError", "prepare_line", "__version__", "RELEASE"]


def __getattr__(name):
    if name in ("HTRJANDModel", "Transcription"):
        from . import hub
        return getattr(hub, name)
    if name in ("NotALineError", "prepare_line"):
        from . import inference
        return getattr(inference, name)
    raise AttributeError(f"module 'htrjand' has no attribute {name!r}")
