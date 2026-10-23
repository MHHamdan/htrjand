"""Release identifiers, in ONE place (release addition).

Renaming the Hugging Face repositories is a one-line change here: every default in
the package, predict.py, app.py, the notebook and scripts/export_hf.py derives from
HF_NAMESPACE and REPO_PREFIX.
"""
import os

HF_NAMESPACE = os.environ.get("HTRJAND_HF_NAMESPACE", "MHamdan")
REPO_PREFIX = os.environ.get("HTRJAND_REPO_PREFIX", "htrjand")

WORK_NAME = "HTR-JAND"                                # how the cards and the demo name the work
WORK_DESCRIPTION = "HTR-JAND"
RELEASE_VERSION = "v1.0"
SOURCE_COMMIT = "v1.0"                               # public release tag of GITHUB_REPO holding this code
PRIVATE_SOURCE_COMMIT = ("ebbfbcd (code and results), a30a640 (audits, controls), 0fc8721 (figures) "
                         "in the private research repository")
GITHUB_REPO = "DocumentRecognitionModels/htrjand"
PAPER_TITLE = "HTR-JAND: Handwritten Text Recognition with Joint Attention Network and Knowledge Distillation"
PAPER_AUTHORS = ("Mohammed Hamdan", "Abderrahmane Rahiche", "Mohamed Cheriet")
PAPER_BIBKEY = "hamdan2026htrjand"
PAPER_NOTE = "manuscript under review"

STUDENT = f"{REPO_PREFIX}-student"
STUDENT_SEED2 = f"{REPO_PREFIX}-student-seed2"
TEACHER = f"{REPO_PREFIX}-teacher"
T5_CORRECTOR = f"{REPO_PREFIX}-t5-corrector"
CHARLM = f"{REPO_PREFIX}-charlm"                     # Wikipedia-derived LM + lexicons (CC BY-SA 4.0)
SPACE = f"{REPO_PREFIX}-demo"                        # the Gradio Space (and its staging folder)

REPOS = (STUDENT, STUDENT_SEED2, TEACHER, T5_CORRECTOR, CHARLM)


def repo_id(name):
    """'htrjand-student' -> 'MHamdan/htrjand-student'."""
    return f"{HF_NAMESPACE}/{name}"


DEFAULT_MODEL = repo_id(STUDENT)
DEFAULT_T5 = repo_id(T5_CORRECTOR)
DEFAULT_CHARLM = repo_id(CHARLM)
VARIANTS = ("general", "iam", "rimes", "bentham", "saintgall", "washington")
