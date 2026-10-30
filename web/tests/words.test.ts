/**
 * The words of the page: run names made readable, and how episodes ended.
 */

import { describe, expect, it } from "vitest";

import { counterpart, ending, label } from "../src/ui/words";

describe("controller names", () => {
  it("say the algorithm, how it was trained, and which seed", () => {
    expect(label("pid")).toBe("PID");
    expect(label("ppo_hover_randomised_s1")).toBe("PPO · randomised · #1");
    expect(label("sac_goto_randomised_s2_best")).toBe("SAC · randomised · #2 · best");
  });

  it("show a transfer and a fine-tuning for what they are", () => {
    expect(label("ppo_navigation_from_obstacles_s5")).toBe("PPO · transfer from obstacles · #5");
    expect(label("ppo_navigation_sparse_from_obstacles_s1")).toBe("PPO · sparse reward, transfer from obstacles · #1");
    expect(label("ppo_goto_finetuned_long_s2")).toBe("PPO · fine-tuned, longer · #2");
    expect(label("ppo_goto_finetuned_s2")).toBe("PPO · fine-tuned · #2");
    expect(label("ppo_navigation_g995_f50_s3")).toBe("PPO · γ 0.995, failure penalty −50 · #3");
  });

  it("keep what they do not know readable", () => {
    expect(label("ppo_hover_new_idea_s4")).toBe("PPO · new idea · #4");
    expect(label("td3_navigation_s1")).toBe("TD3 · #1");
    expect(label("something_else")).toBe("something_else");
  });
});

describe("endings", () => {
  it("in plain words", () => {
    expect(ending("goto", true, "time_limit", true)).toBe("success");
    expect(ending("goto", false, "crash", false)).toBe("hit the ground");
    expect(ending("navigation", false, "no_course", true)).toMatch(/Target tool/);
    expect(ending("goto", false, "some_new_reason", false)).toBe("some new reason");
  });
});

describe("a controller's counterpart on another task", () => {
  const goto = ["pid", "lqr", "ppo_goto_normal_s1", "ppo_goto_normal_s2", "ppo_goto_randomised_s2",
    "sac_goto_randomised_s1", "sac_goto_randomised_s1_best", "td3_goto_normal_s3"];

  it("keeps a controller that is still offered, PID and LQR on every task", () => {
    expect(counterpart("pid", goto)).toBe("pid");
    expect(counterpart("lqr", goto)).toBe("lqr");
    expect(counterpart("ppo_goto_normal_s2", goto)).toBe("ppo_goto_normal_s2");
  });

  it("finds the same algorithm, variant, seed and checkpoint on the new task", () => {
    expect(counterpart("ppo_hover_randomised_s2", goto)).toBe("ppo_goto_randomised_s2");
    expect(counterpart("sac_hover_randomised_s1_best", goto)).toBe("sac_goto_randomised_s1_best");
    expect(counterpart("sac_hover_randomised_s1", goto)).toBe("sac_goto_randomised_s1");
  });

  it("gives up the seed before the variant, and the variant before the algorithm", () => {
    expect(counterpart("ppo_hover_randomised_s3", goto)).toBe("ppo_goto_randomised_s2");
    expect(counterpart("ppo_navigation_from_obstacles_s1", goto)).toBe("ppo_goto_normal_s1");
    expect(counterpart("td3_hover_randomised_s1", goto)).toBe("td3_goto_normal_s3");
  });

  it("has none when no network of that algorithm flies the new task", () => {
    expect(counterpart("td3_hover_normal_s1", ["pid", "lqr", "ppo_obstacles_normal_s1"])).toBeUndefined();
    expect(counterpart("something_else", goto)).toBeUndefined();
  });
});
