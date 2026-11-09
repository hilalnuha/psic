import random
import numpy as np
from collections import defaultdict

def run_psic_aloha_full_simulation(loads, capture_prob=0.8, slots=50000, W=10):
    """
    Simulates Throughput (S), Packet Loss Ratio (PLR), and Average Latency (D)
    for Standard Slotted ALOHA (SA) and PSIC-ALOHA.
    
    Parameters:
        loads (array-like): List/array of offered load (G) values.
        capture_prob (float): Probability beta of successfully recovering packets in PSIC.
        slots (int): Total duration of simulation in slots.
        W (int): Maximum contention window size for random backoff retransmission.
    """
    results = []
    
    for G in loads:
        num_packets = int(G * slots)
        if num_packets == 0:
            continue
            
        # -------------------------------------------------------------
        # Part 1: Throughput (S) & Packet Loss Ratio (PLR) Simulation
        # -------------------------------------------------------------
        # Standard Slotted ALOHA (SA)
        sa_success = 0
        sa_slots = defaultdict(int)
        for _ in range(num_packets):
            sa_slots[random.randint(0, slots - 1)] += 1
        for count in sa_slots.values():
            if count == 1:
                sa_success += 1
                
        # PSIC-ALOHA
        psic_success = 0
        psic_slots = defaultdict(list)
        for _ in range(num_packets):
            slot = random.randint(0, slots - 1)
            power = random.choice(['Low', 'High'])
            psic_slots[slot].append(power)
            
        for transmissions in psic_slots.values():
            count = len(transmissions)
            if count == 1:
                psic_success += 1
            elif count == 2:
                # Power diversity check (One High, One Low)
                if 'High' in transmissions and 'Low' in transmissions:
                    # Probabilistic recovery based on capture_prob (beta)
                    if random.random() < capture_prob:
                        psic_success += 2  # Both recovered
                    else:
                        psic_success += 1  # Only High-power recovered

        sa_thr = sa_success / slots
        psic_thr = psic_success / slots
        sa_plr = (num_packets - sa_success) / num_packets
        psic_plr = (num_packets - psic_success) / num_packets

        # -------------------------------------------------------------
        # Part 2: Latency Simulation (Discrete Retransmission Backoff)
        # -------------------------------------------------------------
        def simulate_latency_protocol(is_psic):
            # Track: [current_attempt_slot, attempt_counter]
            pending_packets = [[random.randint(0, slots - 1), 0] for _ in range(num_packets)]
            successful_delays = []
            
            current_slot = 0
            # Buffer extra slots to allow retransmissions to complete
            max_sim_slots = slots * 2 
            
            while pending_packets and current_slot < max_sim_slots:
                active_this_slot = [p for p in pending_packets if p[0] == current_slot]
                count = len(active_this_slot)
                
                if count == 1:
                    p = active_this_slot[0]
                    successful_delays.append(p[1] + 1)
                    pending_packets.remove(p)
                    
                elif count == 2 and is_psic:
                    powers = random.choices(['Low', 'High'], k=2)
                    if 'High' in powers and 'Low' in powers and random.random() < capture_prob:
                        for p in active_this_slot:
                            successful_delays.append(p[1] + 1)
                            pending_packets.remove(p)
                    else:
                        for p in active_this_slot:
                            p[0] += random.randint(1, W)
                            p[1] += 1
                elif count >= 2:
                    for p in active_this_slot:
                        p[0] += random.randint(1, W)
                        p[1] += 1
                        
                current_slot += 1
                
            return np.mean(successful_delays) if successful_delays else 1.0

        sa_lat = simulate_latency_protocol(is_psic=False)
        psic_lat = simulate_latency_protocol(is_psic=True)

        # Store metrics
        results.append({
            'G': G,
            'SA_S': sa_thr, 'PSIC_S': psic_thr,
            'SA_PLR': sa_plr, 'PSIC_PLR': psic_plr,
            'SA_Lat': sa_lat, 'PSIC_Lat': psic_lat
        })
        
    return results


# -------------------------------------------------------------
# Execute Simulation & Print Formatted LaTeX/Terminal Results
# -------------------------------------------------------------
if __name__ == "__main__":
    # Load range: 0.2 to 2.0 with step 0.2
    load_range = np.arange(0.2, 2.2, 0.2)
    
    print("Running simulation, please wait...")
    sim_data = run_psic_aloha_full_simulation(load_range, capture_prob=0.8, slots=30000, W=10)

    # Output formatted metric table
    header = f"{'Load(G)':<8} | {'SA S':<8} | {'PSIC S':<8} | {'SA PLR':<8} | {'PSIC PLR':<8} | {'SA Lat':<8} | {'PSIC Lat':<8}"
    print("\n" + "=" * len(header))
    print(header)
    print("=" * len(header))
    for r in sim_data:
        print(f"{r['G']:<8.1f} | {r['SA_S']:<8.3f} | {r['PSIC_S']:<8.3f} | {r['SA_PLR']:<8.3f} | {r['PSIC_PLR']:<8.3f} | {r['SA_Lat']:<8.3f} | {r['PSIC_Lat']:<8.3f}")
    print("=" * len(header))
