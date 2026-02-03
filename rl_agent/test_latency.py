
import numpy as np
import time
from rl_zmq_client import RLZmqClient

def main():
    print("Initializing RLZmqClient...")
    # Initialize client (assumes server is running)
    client = RLZmqClient()

    n_steps = 1000
    latencies = []
    sim_steps = 100
    cmd_vel = np.array([0.5, 0.1])
    
    # Calculate Hz assuming 1ms physics step
    update_rate_hz = 1.0 / (sim_steps * 0.001)
    
    print(f"Initializing RLZmqClient...")
    print(f"Update Rate: {update_rate_hz:.2f} Hz (Steps: {sim_steps}, dt: {sim_steps*0.001:.3f}s)")
    print(f"Starting benchmark over {n_steps} requests...")
    
    for i in range(n_steps):
        start_time = time.perf_counter()
        response = client.step(cmd_vel, steps=sim_steps, reset=False)
        end_time = time.perf_counter()
        
        latency_ms = (end_time - start_time) * 1000
        latencies.append(latency_ms)
        
        if (i + 1) % 10 == 0:
            print(f"Completed {i + 1}/{n_steps} requests. Last latency: {latency_ms:.2f} ms")

    latencies = np.array(latencies)
    
    # Calculate Real Time Factor (RTF)
    # RTF = Sim Time Delta / Wall Time Delta
    # Sim Time Delta = steps * 0.001s (assuming 1ms physics step)
    sim_time_delta_ms = sim_steps * 1.0 # 0.001s * 1000 = 1ms per step
    
    # Avoid division by zero
    rtfs = sim_time_delta_ms / (latencies + 1e-6)
    
    # Calculate Overhead Latency (Network + Processing)
    # Total Latency = Sim Time Wait + Overhead
    overhead_latencies = latencies - sim_time_delta_ms

    print("\n--- Latency Report (Total Round Trip) ---")
    print(f"Count: {len(latencies)}")
    print(f"Average: {np.mean(latencies):.2f} ms")
    print(f"Std Dev: {np.std(latencies):.2f} ms")
    
    print("\n--- Overhead Latency (Total - Sim Wait) ---")
    print(f"Sim Time Wait: {sim_time_delta_ms:.2f} ms")
    print(f"Average Overhead: {np.mean(overhead_latencies):.2f} ms")
    print(f"Std Dev: {np.std(overhead_latencies):.2f} ms")
    print(f"Min: {np.min(overhead_latencies):.2f} ms")
    print(f"Max: {np.max(overhead_latencies):.2f} ms")
    print(f"Median: {np.percentile(overhead_latencies, 50):.2f} ms")
    print(f"99%: {np.percentile(overhead_latencies, 99):.2f} ms")
    
    print("\n--- Real Time Factor (RTF) ---")
    print("RTF = (Sim Time / Wall Time). 1.0 = Real Time. >1.0 = Faster than Real Time.")
    print(f"Average RTF: {np.mean(rtfs):.2f}x")
    print(f"Median RTF: {np.percentile(rtfs, 50):.2f}x")

    # Save Histogram of OVERHEAD
    try:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(10, 6))
        plt.hist(overhead_latencies, bins=50, color='orange', alpha=0.7, edgecolor='black')
        plt.title(f"ZeroMQ Bridge Latency Distribution (n={len(latencies)}, steps={sim_steps})")
        plt.xlabel("Latency (ms)")
        plt.ylabel("Frequency")
        plt.axvline(np.mean(overhead_latencies), color='red', linestyle='dashed', linewidth=1, label=f'Mean: {np.mean(overhead_latencies):.2f}ms')
        plt.axvline(np.median(overhead_latencies), color='green', linestyle='dashed', linewidth=1, label=f'Median: {np.median(overhead_latencies):.2f}ms')
        plt.legend()
        plt.grid(True, alpha=0.3)
        output_file = "overhead_latency_histogram.png"
        plt.savefig(output_file)
        print(f"\nHistogram saved to {output_file}")
    except ImportError:
        print("\nMatplotlib not installed. Skipping histogram.")


if __name__ == "__main__":
    main()
