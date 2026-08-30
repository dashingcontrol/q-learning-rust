use rand::rngs::StdRng;
use rand::{RngExt, SeedableRng};
use std::fs::{create_dir_all, write};

// 6x6 maze
const ROWS: usize = 6;
const COLS: usize = 6;
const ACTIONS: usize = 4;
const START: (usize, usize) = (5, 0);
const GOAL: (usize, usize) = (0, 5);
const EPISODES: usize = 100;
const MAX_STEPS: usize = 200;

// 0: Up, 1: Down, 2: Left, 3: Right
const ACTION_NAMES: [&str; ACTIONS] = ["U", "D", "L", "R"];

// walls.
const WALLS: [(usize, usize); 11] = [
    (0, 3),
    (1, 1),
    (1, 3),
    (2, 1),
    (2, 5),
    (3, 1),
    (3, 2),
    (3, 3),
    (4, 5),
    (5, 1),
    (5, 3),
];

// QTable structure: [row][col][action]
type QTable = [[[f64; ACTIONS]; COLS]; ROWS];

fn is_wall(r: usize, c: usize) -> bool { 
    WALLS.contains(&(r, c))
}

fn best_action(q: &QTable, state: (usize, usize)) -> usize {
    let values = q[state.0][state.1]; 
    let mut best_idx = 0;
    for i in 1..ACTIONS {
        if values[i] > values[best_idx] {
            best_idx = i;
        }
    }
    best_idx
}

fn best_value(q: &QTable, state: (usize, usize)) -> f64 {
    let mut max_val = f64::NEG_INFINITY;
    for &val in &q[state.0][state.1] {
        if val > max_val {
            max_val = val;
        }
    }
    max_val
}

fn choose_action(q: &QTable, state: (usize, usize), epsilon: f64, rng: &mut StdRng) -> usize {
    if rng.random::<f64>() < epsilon {
        rng.random_range(0..ACTIONS)
    } else {
        best_action(q, state)
    }
}

fn move_agent(state: (usize, usize), action: usize) -> (usize, usize) {
    let (r, c) = state;
    
    let next_pos = match action {
        0 if r > 0 => (r - 1, c),        // Up
        1 if r + 1 < ROWS => (r + 1, c), // Down
        2 if c > 0 => (r, c - 1),        // Left
        3 if c + 1 < COLS => (r, c + 1), // Right
        _ => (r, c), // Hit a boundary, stay put
    };

    // Check if the new spot is a wall
    if is_wall(next_pos.0, next_pos.1) {
        state
    } else {
        next_pos
    }
}


fn save_q_table(filename: &str, q: &QTable) {
    let mut csv = String::from("row,col,up,down,left,right,best_action,best_q\n");

    for r in 0..ROWS {
        for c in 0..COLS {
            if is_wall(r, c) {
                continue;
            }

            let state = (r, c);
            let action = best_action(q, state);
            
            // Just some logic to make the CSV readable
            let action_label = if state == GOAL {
                "G"
            } else if q[r][c].iter().all(|v| v.abs() < 1e-12) {
                "-"
            } else {
                ACTION_NAMES[action]
            };
            
            csv.push_str(&format!(
                "{r},{c},{:.4},{:.4},{:.4},{:.4},{},{:.4}\n",
                q[r][c][0],
                q[r][c][1],
                q[r][c][2],
                q[r][c][3],
                action_label,
                best_value(q, state)
            ));
        }
    }

    write(filename, csv).expect("Failed to write the Q-table file. Check permissions?");
}

fn save_trajectory(filename: &str, path: &[(usize, usize)]) {
    let mut csv = String::from("step,row,col\n");
    for (i, &(r, c)) in path.iter().enumerate() {
        csv.push_str(&format!("{i},{r},{c}\n"));
    }
    write(filename, csv).expect("Failed to save trajectory.");
}


fn print_policy(title: &str, q: &QTable) {
    println!("\n--- {} ---", title);
    for r in 0..ROWS {
        for c in 0..COLS {
            let state = (r, c);
            let display = if is_wall(r, c) {
                "#"
            } else if state == START {
                "S"
            } else if state == GOAL {
                "G"
            } else if q[r][c].iter().all(|v| v.abs() < 1e-12) {
                "."
            } else {
                ACTION_NAMES[best_action(q, state)]
            };
            print!("{display:>2} ");
        }
        println!();
    }
}

fn main() {
    // parameters
    let alpha = 0.40; // learning rate
    let gamma = 0.90; // discount factor
    let mut rng = StdRng::seed_from_u64(7);

    // Initialize Q-table with zeros
    let mut q: QTable = [[[0.0; ACTIONS]; COLS]; ROWS];
    
    // Keeping copies to see how it changes over time
    let q_before = q;
    let mut q_middle = q;
    
    let mut episode_data = String::from("episode,steps,epsilon,reached_goal\n");
    let mut trajectories_to_save: Vec<(usize, Vec<(usize, usize)>)> = Vec::new();

    // Main training loop
    for ep in 1..=EPISODES {
        // Epsilon decay: start high, get lower
        let eps = (0.40 * 0.97_f64.powi((ep - 1) as i32)).max(0.02);
        
        let mut curr = START;
        let mut path = vec![curr];

        for _ in 0..MAX_STEPS {
            let act = choose_action(&q, curr, eps, &mut rng);
            let next = move_agent(curr, act);
            
            // Reward is -1 for every step to encourage shortest path
            let reward = -1.0; 

            // Q-learning update rule
            // Q(s,a) = Q(s,a) + alpha * (reward + gamma * max(Q(s',a')) - Q(s,a))
            let target = if next == GOAL {
                0.0
            } else {
                gamma * best_value(&q, next)
            };
            
            let old_q = q[curr.0][curr.1][act];
            q[curr.0][curr.1][act] = old_q + alpha * (reward + target - old_q);

            curr = next;
            path.push(curr);
            
            if curr == GOAL {
                break;
            }
        }
       
        let steps = path.len() - 1;
        let reached = curr == GOAL;
        episode_data.push_str(&format!("{ep},{steps},{eps:.4},{reached}\n"));

        if ep == 1 || ep == 10 || ep == 100 {
            trajectories_to_save.push((ep, path));
        }
          
        if ep == 50 {
            q_middle = q;
        }
    }

    // Save everything to files
    create_dir_all("output/data").expect("Couldn't make output dir");
    write("output/data/episodes.csv", episode_data).expect("Couldn't save episode data");
    
    save_q_table("output/data/q_before.csv", &q_before);
    save_q_table("output/data/q_middle.csv", &q_middle);
    save_q_table("output/data/q_after.csv", &q);

    // Save trajectories
    for (ep, path) in &trajectories_to_save {
        save_trajectory(&format!("output/data/trajectory_{ep}.csv"), path);
        println!("Episode {ep:>3}: {steps:>3} steps, goal reached: {}", 
            path.len() - 1, 
            path.last() == Some(&GOAL)
        );
    }

    print_policy("Policy before learning:", &q_before);
    print_policy("Greedy policy after episode 50:", &q_middle);
    print_policy("Greedy policy after episode 100:", &q);
    
    println!("\nDone.");
}

