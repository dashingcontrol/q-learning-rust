use rand::rngs::StdRng;
use rand::{RngExt, SeedableRng};
use std::fs::{create_dir_all, write};

const ROWS: usize = 6;
const COLS: usize = 6;
const ACTIONS: usize = 4;
const START: (usize, usize) = (5, 0);
const GOAL: (usize, usize) = (0, 5);
const EPISODES: usize = 100;
const MAX_STEPS: usize = 200;
const ACTION_NAMES: [&str; ACTIONS] = ["U", "D", "L", "R"];

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

type QTable = [[[f64; ACTIONS]; COLS]; ROWS];

fn is_wall(state: (usize, usize)) -> bool {
    WALLS.contains(&state)
}

// invalid moves stay in the same cell.
fn move_agent(state: (usize, usize), action: usize) -> (usize, usize) {
    let (row, col) = state;
    let next = match action {
        0 if row > 0 => (row - 1, col),
        1 if row + 1 < ROWS => (row + 1, col),
        2 if col > 0 => (row, col - 1),
        3 if col + 1 < COLS => (row, col + 1),
        _ => state,
    };
    if is_wall(next) { state } else { next }
}

fn legal_actions(state: (usize, usize)) -> Vec<usize> {
    (0..ACTIONS)
        .filter(|&action| move_agent(state, action) != state)
        .collect()
}

fn best_action(q: &QTable, state: (usize, usize)) -> usize {
    let legal = legal_actions(state);
    let mut best = legal[0];
    for &action in &legal[1..] {
        if q[state.0][state.1][action] > q[state.0][state.1][best] {
            best = action;
        }
    }
    best
}

fn best_value(q: &QTable, state: (usize, usize)) -> f64 {
    q[state.0][state.1][best_action(q, state)]
}

// best actions are chosen randomly during training.
fn random_best_action(q: &QTable, state: (usize, usize), rng: &mut StdRng) -> usize {
    let legal = legal_actions(state);
    let best = legal
        .iter()
        .map(|&action| q[state.0][state.1][action])
        .fold(f64::NEG_INFINITY, f64::max);
    let tied: Vec<usize> = legal
        .into_iter()
        .filter(|&action| (q[state.0][state.1][action] - best).abs() < 1e-12)
        .collect();
    tied[rng.random_range(0..tied.len())]
}

fn choose_action(q: &QTable, state: (usize, usize), epsilon: f64, rng: &mut StdRng) -> usize {
    let legal = legal_actions(state);
    if rng.random::<f64>() < epsilon {
        legal[rng.random_range(0..legal.len())]
    } else {
        random_best_action(q, state, rng)
    }
}

fn update_q(old_q: f64, reward: f64, future: f64, alpha: f64) -> f64 {
    old_q + alpha * (reward + future - old_q)
}

fn save_trajectory(filename: &str, path: &[(usize, usize)]) {
    let mut csv = String::from("step,row,col\n");
    for (step, &(row, col)) in path.iter().enumerate() {
        csv.push_str(&format!("{step},{row},{col}\n"));
    }
    write(filename, csv).expect("could not write trajectory");
}

fn save_q_table(filename: &str, q: &QTable) {
    let mut csv = String::from("row,col,up,down,left,right,best_action,best_q\n");
    for row in 0..ROWS {
        for col in 0..COLS {
            let state = (row, col);
            if is_wall(state) {
                continue;
            }
            let action = best_action(q, state);
            let label = if state == GOAL {
                "G"
            } else if legal_actions(state)
                .iter()
                .all(|&a| q[row][col][a].abs() < 1e-12)
            {
                "-"
            } else {
                ACTION_NAMES[action]
            };
            csv.push_str(&format!(
                "{row},{col},{:.4},{:.4},{:.4},{:.4},{label},{:.4}\n",
                q[row][col][0],
                q[row][col][1],
                q[row][col][2],
                q[row][col][3],
                best_value(q, state)
            ));
        }
    }
    write(filename, csv).expect("could not write Q-table");
}

//read-only test of the final policy: epsilon = 0 and no updates.
fn evaluate_greedy(q: &QTable) -> Vec<(usize, usize)> {
    let mut state = START;
    let mut path = vec![state];
    for _ in 0..MAX_STEPS {
        state = move_agent(state, best_action(q, state));
        path.push(state);
        if state == GOAL {
            break;
        }
    }
    path
}

fn main() {
    create_dir_all("output/data").expect("could not create output/data");

    let alpha = 0.40;
    let gamma = 0.90;
    let mut rng = StdRng::seed_from_u64(7);
    let mut q: QTable = [[[0.0; ACTIONS]; COLS]; ROWS];
    let q_before = q;
    let mut q_middle = q;
    let mut episodes_csv = String::from("episode,steps,epsilon,reached_goal\n");
    let mut saved_paths = Vec::new();

    for episode in 1..=EPISODES {
        let epsilon = (0.40 * 0.97_f64.powi((episode - 1) as i32)).max(0.02);
        let mut state = START;
        let mut path = vec![state];

        for _ in 0..MAX_STEPS {
            let action = choose_action(&q, state, epsilon, &mut rng);
            let next = move_agent(state, action);
            let future = if next == GOAL {
                0.0
            } else {
                gamma * best_value(&q, next)
            };

            let old_q = q[state.0][state.1][action];
            q[state.0][state.1][action] = update_q(old_q, -1.0, future, alpha);
            state = next;
            path.push(state);
            if state == GOAL {
                break;
            }
        }

        episodes_csv.push_str(&format!(
            "{episode},{},{epsilon:.4},{}\n",
            path.len() - 1,
            state == GOAL
        ));
        if [1, 10, 100].contains(&episode) {
            saved_paths.push((episode, path));
        }
        if episode == 50 {
            q_middle = q;
        }
    }

    write("output/data/episodes.csv", episodes_csv).expect("could not write episodes.csv");
    save_q_table("output/data/q_before.csv", &q_before);
    save_q_table("output/data/q_middle.csv", &q_middle);
    save_q_table("output/data/q_after.csv", &q);
    for (episode, path) in &saved_paths {
        save_trajectory(&format!("output/data/trajectory_{episode}.csv"), path);
        println!("Episode {episode:>3}: {} steps", path.len() - 1);
    }

    let greedy_path = evaluate_greedy(&q);
    let reached_goal = greedy_path.last() == Some(&GOAL);
    save_trajectory("output/data/trajectory_greedy.csv", &greedy_path);
    write(
        "output/data/greedy_evaluation.csv",
        format!(
            "steps,reached_goal\n{},{}\n",
            greedy_path.len() - 1,
            reached_goal
        ),
    )
    .expect("could not write greedy evaluation");
    println!(
        "Greedy evaluation: {} steps, reached goal = {reached_goal}",
        greedy_path.len() - 1
    );
}