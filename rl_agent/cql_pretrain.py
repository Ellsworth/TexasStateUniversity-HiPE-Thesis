import d3rlpy
import argparse
import os

from d3rlpy.logging import TensorboardAdapterFactory

from firebot_agent.utils import convert_continuous_to_discrete
from firebot_agent.log_master import FireBotLogger
from firebot_agent.training_utils import (
    print_gpu_info,
    load_dataset,
    create_cql,
    get_evaluators,
)


def main():
    parser = argparse.ArgumentParser(description="DiscreteCQL Offline Pretraining")
    parser.add_argument("--dataset", type=str, default="offline_dataset.npz", help="Path to the offline dataset (npz file or directory of npz files)")
    parser.add_argument("--pretrain-steps", type=int, default=1000000, help="Number of offline pretraining steps")
    parser.add_argument("--n-frames", type=int, default=4, help="Number of frames to stack")

    args = parser.parse_args()

    # Create a single timestamped log directory for this run
    logger = FireBotLogger(base_dir="logs", experiment_name="CQL_Pretrain")
    log_dir = logger.get_log_dir()

    # Output model path
    model_path = os.path.join(log_dir, "cql_pretrained.d3")

    print_gpu_info()

    cql = create_cql()
    evaluators = get_evaluators()

    # Load dataset
    print("=" * 60)
    print("OFFLINE PRETRAINING WITH DISCRETECQL")
    print("=" * 60)

    observations, actions_raw, rewards, terminals = load_dataset(args.dataset)

    print("Total observations shape:", observations.shape)

    actions = convert_continuous_to_discrete(actions_raw)

    transition_picker = None
    if args.n_frames > 1:
        print(f"Using FrameStackTransitionPicker with n_frames={args.n_frames}")
        transition_picker = d3rlpy.dataset.FrameStackTransitionPicker(n_frames=args.n_frames)

    dataset = d3rlpy.dataset.MDPDataset(
        observations=observations,
        actions=actions,
        rewards=rewards,
        terminals=terminals,
        transition_picker=transition_picker
    )

    print(f"Training DiscreteCQL for {args.pretrain_steps} steps...")
    cql.fit(
        dataset,
        n_steps=args.pretrain_steps,
        experiment_name="CQL_Offline_Pretrain",
        logger_adapter=TensorboardAdapterFactory(root_dir=log_dir),
        evaluators=evaluators,
    )

    # Save pretrained model
    cql.save(model_path)
    print(f"\nPretrained model saved to {model_path}")


if __name__ == "__main__":
    main()
