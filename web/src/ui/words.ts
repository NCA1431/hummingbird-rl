/**
 * The words the page uses for tasks, controllers and endings.
 *
 * Kept apart from the panel (which needs a page to run) so that the tests can check them.
 */

export const TASK_NAMES: Record<string, string> = {
  hover: "Hover & recover", goto: "Go-to-point", obstacles: "Obstacles", navigation: "Navigation",
};

/**
 * How an episode ended, in plain words: short for the menus, long for the HUD. The limits are
 * those of the simulator: ground at 2 cm, arena 1.5 m wide and 1.2 m high, 80° of pitch or roll.
 */
export function ending(task: string, success: boolean, reason: string, long: boolean): string {
  if (success) return "success";
  const words: Record<string, [string, string]> = {
    time_limit: [
      "time limit",
      task === "hover"
        ? "time limit: more than 3 cm from the target on average over the last 2 s"
        : "time limit: the target was not held in time",
    ],
    loss_of_control: ["lost control", "lost control: tilted past 80°"],
    collision: ["hit an obstacle", "hit an obstacle"],
    crash: ["hit the ground", "crashed into the ground"],
    out_of_bounds: ["left the arena", "left the arena"],
    no_course: ["no course yet", "no course yet: place the waypoints with the Target tool"],
  };
  const w = words[reason];
  return w ? w[long ? 1 : 0] : reason.replace(/_/g, " ");
}

/**
 * The parts of a run name after its task, in words, longest first so that "finetuned_long" is
 * not read as "finetuned" followed by "long". A transfer is said as such: the network started
 * from one trained on another task, and the menu should show it.
 */
const VARIANTS: [string, string][] = [
  ["sparse_from_obstacles", "sparse reward, transfer from obstacles"],
  ["from_obstacles", "transfer from obstacles"],
  ["finetuned_long", "fine-tuned, longer"],
  ["finetuned", "fine-tuned"],
  ["randomised", "randomised"],
  ["normal", "normal"],
  ["sparse", "sparse reward"],
  ["anneal", "annealed learning rate"],
  ["f50", "failure penalty −50"],
  ["g995", "γ 0.995"],
  ["lambda0", "λ 0"],
  ["lambda1", "λ 1"],
  ["noadvnorm", "no advantage normalisation"],
  ["noclip", "no clipping"],
  ["nohoverinit", "no hover start"],
  ["fixedalpha", "fixed α"],
];

function variantWords(variant: string): string {
  const words: string[] = [];
  let rest = variant;
  while (rest) {
    const known = VARIANTS.find(([k]) => rest === k || rest.startsWith(`${k}_`));
    if (!known) {
      words.push(rest.replace(/_/g, " ")); // a variant we do not know: its name, readable
      break;
    }
    words.push(known[1]);
    rest = rest.slice(known[0].length + 1);
  }
  return words.join(", ");
}

/**
 * "PID", "LQR", or a run name made readable:
 *   ppo_hover_randomised_s1              -> "PPO · randomised · #1"
 *   ppo_navigation_from_obstacles_s5     -> "PPO · transfer from obstacles · #5"
 *   sac_goto_randomised_s2_best          -> "SAC · randomised · #2 · best"
 * "best" is the checkpoint with the best validation score, not the last one.
 */
/** A run name: algorithm, task, training variant, seed and, for the best checkpoint, _best. */
const RUN = /^(ppo|sac|td3)_(hover|goto|obstacles|navigation)(?:_(.+?))?_s(\d+)(_best)?$/;

export function label(p: string): string {
  if (p === "pid" || p === "lqr") return p.toUpperCase();
  const m = RUN.exec(p);
  if (!m) return p;
  return [m[1].toUpperCase(), m[3] ? variantWords(m[3]) : "", `#${m[4]}`, m[5] ? "best" : ""].filter(Boolean).join(" · ");
}

/**
 * The controller that plays the same part among `offered`, when the task changes and `name` is no
 * longer there: networks are trained for one task, so "PPO · randomised · #2" on hover becomes
 * ppo_goto_randomised_s2 on go-to-point. Same algorithm required; then, in this order of
 * importance, the same training variant, the same seed and the same checkpoint (final or best).
 * PID and LQR fly every task. Undefined when no network of that algorithm is offered.
 */
export function counterpart(name: string, offered: string[]): string | undefined {
  if (offered.includes(name)) return name;
  const parts = (n: string) => {
    const m = RUN.exec(n);
    return m ? { algo: m[1], variant: m[3] ?? "", seed: m[4], best: Boolean(m[5]) } : null;
  };
  const want = parts(name);
  if (!want) return undefined;
  let found: string | undefined;
  let score = -1;
  for (const n of offered) {
    const got = parts(n);
    if (!got || got.algo !== want.algo) continue;
    const s = 4 * Number(got.variant === want.variant) + 2 * Number(got.seed === want.seed) + Number(got.best === want.best);
    if (s > score) [found, score] = [n, s];
  }
  return found;
}
