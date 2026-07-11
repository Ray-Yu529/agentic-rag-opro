"""全模組 import 冒煙測試: 沒有循環引用、沒有語法錯，判官與生成模型不同。"""


def test_all_modules_import():
    import cache            # noqa: F401
    import dataset          # noqa: F401
    import eval as ev       # noqa: F401
    import hybrid           # noqa: F401
    import judge
    import optimizer        # noqa: F401
    import pareto           # noqa: F401
    import plot             # noqa: F401
    import rag
    import run              # noqa: F401
    import server           # noqa: F401
    import sweep            # noqa: F401
    from memory import trajectory, warmstart  # noqa: F401

    assert judge.JUDGE_MODEL != rag.GEN_MODEL
