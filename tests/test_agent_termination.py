from unittest.mock import MagicMock
from finrl.benchmark.agent import QwenAgent
from finrl.env.rule_605_env import Rule605Env


def test_agent_terminates_on_submit_report():
    # Setup mock env
    env = MagicMock(spec=Rule605Env)
    
    # We want env.done to be False initially to let the loop start
    env.done = False
    env.current_step = 0
    env.max_steps = 10
    env.ground_truth_pipe = "mock_pipe"
    
    obs_mock = MagicMock()
    obs_mock.scenario_id = "test_123"
    obs_mock.security = "FINRL"
    obs_mock.orders = []
    obs_mock.model_dump.return_value = {}
    env._get_observation.return_value = obs_mock

    # Step result mock
    step_res_mock = MagicMock()
    step_res_mock.observation = obs_mock
    step_res_mock.info = {"tool_output": "report submitted"}
    step_res_mock.reward = 0.0
    env.step.return_value = step_res_mock
    
    # Mock runner
    runner_mock = MagicMock()
    # Return submit_report immediately
    runner_mock.generate.return_value = (
        'Thought: done.\nAction: {"tool_name": "submit_report", "arguments": {"report": "done"}}',
        10, 10, 100.0
    )
    
    agent = QwenAgent(runner=runner_mock)
    agent.traces_dir = MagicMock()  # Mock save_trace dir to avoid writing to disk
    agent.run(env)
    
    # Assert runner.generate was called EXACTLY ONCE
    assert runner_mock.generate.call_count == 1
    # Assert env.step was called EXACTLY ONCE
    assert env.step.call_count == 1
