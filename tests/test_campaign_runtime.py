"""Offline campaign checks through the real agent, file tools and scorers."""

import csv
import io
import json
import shutil
from dataclasses import asdict
from types import SimpleNamespace

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import Field

pytest.importorskip("inspect_ai")
from inspect_ai.scorer import Target
from inspect_ai.util import Store

from jutul_agent.agent.builder import build_agent
from jutul_agent.eval.archive import sha256, write_json
from jutul_agent.eval.campaign_validation import CSV_COLUMNS, LEDGER_COLUMNS
from jutul_agent.eval.campaigns import CAMPAIGNS
from jutul_agent.eval.comparisons import CONDITIONS, comparison_task
from jutul_agent.eval.runconfig import julia_environment_sha256
from jutul_agent.eval.tasks._campaign import CAMPAIGN_CASES
from jutul_agent.julia.session import EvalResult
from jutul_agent.lab.fakes import FakeJulia, ScriptedChatModel, scripted_final, scripted_tool_call
from jutul_agent.session import Session
from jutul_agent.simulators.registry import get as simulator


class OfflineCampaignModel(ScriptedChatModel):
    max_tokens: int = 2048
    summary_calls: int = 0
    main_calls: int = 0
    seen_results: list[str] = Field(default_factory=list)

    def _generate(self, messages, *args, **kwargs):
        if messages[0].type != "system":
            self.summary_calls += 1
            return ChatResult(
                generations=[
                    ChatGeneration(
                        message=AIMessage(
                            content=(
                                "The current campaign state is in campaign-index.csv. "
                                "Run remaining cases once, read the complete ledger, "
                                "then grep the selected cell."
                            )
                        )
                    )
                ]
            )
        self.main_calls += 1
        assert self.main_calls <= len(self.responses), "Unexpected extra model request"
        self.seen_results.extend(str(m.content) for m in messages if isinstance(m, ToolMessage))
        return super()._generate(messages, *args, **kwargs)


def csv_text(columns, rows):
    stream = io.StringIO()
    writer = csv.DictWriter(stream, fieldnames=columns, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue()


@pytest.mark.parametrize("campaign", CAMPAIGNS, ids=lambda c: c.name)
@pytest.mark.parametrize("preload", [False, True], ids=["fresh", "long-history"])
@pytest.mark.parametrize(
    "strategies", CONDITIONS, ids=lambda s: f"{s.summarization}-{s.clear_tool_results}"
)
async def test_full_campaign_loop_with_compaction_and_real_file_retrieval(
    tmp_path, monkeypatch, campaign, strategies, preload
):
    monkeypatch.chdir(tmp_path)
    fixture = campaign.fixture()
    (tmp_path / "campaign.jl").write_text(fixture)
    artifacts = {}
    winner = CAMPAIGN_CASES[1] if campaign.query_cell_id == 150 else CAMPAIGN_CASES[0]
    for index, (case, (porosity, rate)) in enumerate(
        zip(CAMPAIGN_CASES, campaign.parameters, strict=True)
    ):
        summary = dict(
            zip(
                CSV_COLUMNS[:6],
                (
                    case,
                    str(porosity),
                    str(rate),
                    str(0.55 if case == winner else 0.4 + index / 100),
                    str(260 + index * 5),
                    str(260 + index * 5),
                ),
                strict=True,
            )
        )
        artifacts[case] = [
            {
                **summary,
                "cell_id": str(cell),
                "cell_pressure_bar": summary["max_pressure_bar"],
                "water_saturation": str(0.8 if cell == campaign.query_cell_id else 0.4),
            }
            for cell in range(1, 193)
        ]
    truth = {
        "run_id": winner,
        "recovery_factor": 0.55,
        "max_pressure_bar": float(artifacts[winner][0]["max_pressure_bar"]),
        "cell_water_saturation": 0.8,
    }
    reference = tmp_path / "reference"
    (reference / "results").mkdir(parents=True)
    (reference / "campaign.jl").write_text(fixture)
    project = tmp_path / "environment"
    project.mkdir()
    (project / "Manifest.toml").write_text("# Offline test fixture\n")
    (project / "Project.toml").write_text("[deps]\n")
    shutil.copytree(project, reference / "julia-environment")
    for case, rows in artifacts.items():
        (reference / f"results/{case}.csv").write_text(csv_text(CSV_COLUMNS, rows))
    write_json(
        reference / "reference.json",
        {
            "campaign": asdict(campaign),
            "fixture_sha256": sha256(reference / "campaign.jl"),
            "manifest_sha256": sha256(project / "Manifest.toml"),
            "julia_environment_sha256": julia_environment_sha256(project),
            "answer": truth,
            "artifact_sha256": {
                case: sha256(reference / f"results/{case}.csv") for case in artifacts
            },
        },
    )
    task = comparison_task(campaign, reference, project, strategies, 1)
    sample = task.dataset[0]
    completed = []

    def julia(code):
        if code == 'include("campaign.jl")':
            return EvalResult(output="campaign loaded")
        case = code.removeprefix('run_autonomous_campaign_case("').removesuffix('")')
        assert case in artifacts and case not in completed
        completed.append(case)
        (tmp_path / "results").mkdir(exist_ok=True)
        (tmp_path / f"results/{case}.csv").write_text(csv_text(CSV_COLUMNS, artifacts[case]))
        ledger = [
            {
                **{key: artifacts[c][0][key] for key in CSV_COLUMNS[:6]},
                "artifact_path": f"results/{c}.csv",
            }
            for c in completed
        ]
        (tmp_path / "campaign-index.csv").write_text(csv_text(LEDGER_COLUMNS, ledger))
        return EvalResult(
            output="step diagnostics\n" * 160 + f"saved results/{case}.csv and campaign-index.csv"
        )

    responses = [scripted_tool_call(tool_name="run_julia", args={"code": 'include("campaign.jl")'})]
    responses += [
        scripted_tool_call(
            tool_name="run_julia", args={"code": f'run_autonomous_campaign_case("{case}")'}
        )
        for case in CAMPAIGN_CASES
    ]
    responses += [
        scripted_tool_call(tool_name="read_file", args={"file_path": "campaign-index.csv"}),
        scripted_tool_call(
            tool_name="grep",
            args={
                "path": f"results/{winner}.csv",
                "pattern": f",{campaign.query_cell_id},",
                "output_mode": "content",
            },
        ),
        scripted_final(json.dumps(truth)),
    ]
    model = OfflineCampaignModel(responses=responses, profile={"max_input_tokens": 16384})
    session = Session.create(
        julia=FakeJulia(eval_handler=julia),
        state_root=tmp_path / "state",
        simulator=simulator("jutuldarcy"),
    )
    try:
        agent, _ = build_agent(session, model=model, harness_strategies=strategies)
        history = []
        for _ in range(8 if preload else 0):
            history.extend(
                [
                    HumanMessage(content="Earlier unrelated work " * 150),
                    AIMessage(content="Archived detail " * 150),
                ]
            )
        result = await agent.ainvoke({"messages": [*history, HumanMessage(content=sample.input)]})
        assert completed == list(CAMPAIGN_CASES)
        assert model.main_calls == len(responses)
        if preload or not strategies.clear_tool_results:
            assert model.summary_calls > 0
        assert len(result["messages"]) > len(history)
        if preload:
            assert any(
                "Archived detail" in m.content
                for m in result["messages"]
                if isinstance(m, AIMessage)
            )
        assert any(
            winner in text and f",{campaign.query_cell_id}," in text for text in model.seen_results
        ), model.seen_results[-2:]
        store = Store()
        store.set("jutul/trace_db", str(session.state_dir / "trace.sqlite"))
        store.set("jutul/workspace", str(tmp_path))
        state = SimpleNamespace(
            store=store,
            metadata=sample.metadata,
            output=SimpleNamespace(completion=result["messages"][-1].content),
        )
        for scorer in task.scorer:
            score = await scorer(state, Target(json.dumps(truth)))
            if isinstance(score.value, str):
                assert score.value == "C", score.explanation
    finally:
        session.trace.close()
