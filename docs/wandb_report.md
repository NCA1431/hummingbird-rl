<!-- title: Teaching a simulated hummingbird to fly: deep RL against classical control -->
<!-- description: PPO, SAC and TD3 against a PID and an LQR on an unstable flapping bird, in gusty air and with damaged wings: four tasks, seven experiments, an interactive demo. -->
<!-- The text of the W&B report. scripts/make_report_wandb.py turns it into the report: "#" lines become
     headings, a "$$" paragraph a LaTeX block, the other paragraphs Markdown blocks, and each "panels:"
     comment the charts it names. -->

A simulated hummingbird, unstable in hover and flying in gusty air, is controlled by three deep
reinforcement learning algorithms (PPO, SAC, TD3) and by two classical controllers (a cascaded PID and an
LQR), on four tasks of increasing difficulty: hover and recover, go to a point, fly around obstacles,
follow a course.

**Links:** [interactive demo](https://nca1431.github.io/hummingbird-rl/) (the bird flies live in your
browser, with every controller of this report) · [code on GitHub](https://github.com/NCA1431/hummingbird-rl)
· [the algorithms explained (PDF)](https://github.com/NCA1431/hummingbird-rl/blob/main/docs/algorithms_report.pdf)

*I chose this problem, defined the tasks and the experiments, ran all 124 training runs and the evaluations
on my own computer, and interpreted the results. I used Claude (Anthropic) as a tool to write most of the
code and to draft the text, which I reviewed and corrected. The end of the report says how the work was
shared and what I learned.*

> **Key findings**
> - **Classical control is hard to beat in calm air.** PID and LQR succeed in 100% of the hover and go-to-point tests, and the PID, thanks to its integrators, is the most robust controller to wind and wing damage (robust area 0.82 out of 1).
> - **Learning pays off where planning matters.** Around obstacles SAC succeeds in 98% and PPO in 89% of the episodes, against 44 to 52% for the classical controllers. On the long navigation course the cautious PID stays best (41%, PPO 18%, 27% with transfer).
> - **Train in randomised conditions.** Trained in calm air, every learned policy loses to the PID under wind (robust area 0.08 to 0.39). Trained with random wind, mass and wing damage, PPO reaches 0.96 on go-to-point, and after a sudden 40% wing failure it deviates by 1.5 cm, against 9.1 cm for the PID.
> - **PPO is the most reliable learner; SAC and TD3 the most sample-efficient but unstable:** their success swings during training (SAC ends at 64%, its best checkpoint scored 99%).

> **What we built and ran**
> - **124 training runs** (102 reported, the others pilots and tuning), **181 million environment steps**: about 840 hours, or 35 days, of simulated flight, computed in about 200 CPU-hours on one desktop PC.
> - **122 controllers evaluated** on up to five test suites: about **78,000 evaluation episodes**, drawn from held-out seeds that every controller faces in the same order.
> - **A simulator and an interactive demo:** the physics in Python with 144 tests, ported to the browser where it reproduces the Python results to 4·10⁻¹² (105 more tests), with a live 3D simulation of the air around the bird.

<!-- toc -->

# 1. Introduction

A hummingbird is a remarkable control problem. It weighs a few grams, beats its wings 40 times a second,
and is the only bird that can hover for long, fly backwards and turn on the spot. Hovering is also
unstable: without constant correction, a small push grows into an oscillation that tips the bird over in
about a second. Add gusts, a heavier bird after a meal, or a damaged feather, and holding still becomes
hard.

We ask five questions:

- **RQ1.** Can deep RL learn to fly such an unstable flapping body, on four tasks of increasing difficulty?
- **RQ2.** How do PPO, SAC and TD3 compare in sample efficiency, stability across seeds and final performance, and against a PID and an LQR?
- **RQ3.** Does RL, in particular when trained in randomised conditions, bring robustness to wind, mass and wing damage that the classical controllers lack?
- **RQ4.** Does knowledge transfer across tasks (Obstacles → Navigation) and across conditions (calm → disturbed)?
- **RQ5.** What does a learned critic represent, compared with the LQR's exact value function?

**Contributions.** A hummingbird simulator with four tasks and three disturbance regimes; five controllers
evaluated on the same held-out episodes; seven experiments (benchmark, flight behaviour, robustness,
fine-tuning, ablations, transfer, interpretability) with 3 to 5 seeds each; and an interactive web demo in
which every controller flies live, with the air around the bird simulated.

# 2. Related work

Flappy Hummingbird (Fei et al. 2019) simulates flapping-wing robots and animals with resolved wing
aerodynamics; we average the wings over each wingbeat to train quickly, and keep their measured motion for
the animation. The flybody fruit fly (Vaxenburg et al. 2025) shows what a full-body model in MuJoCo makes
possible, the natural next step for this project. Our wing kinematics come from high-speed recordings of a
calliope hummingbird (Song et al. 2016). Training under random conditions to obtain robust policies is
domain randomisation (Tobin et al. 2017). The algorithms are PPO (Schulman et al. 2017), TD3 (Fujimoto et
al. 2018) and SAC (Haarnoja et al. 2018), in their Stable-Baselines3 implementations (Raffin et al. 2021),
and the evaluation follows the advice of Henderson et al. (2018) and Agarwal et al. (2021): several seeds,
held-out test episodes, intervals rather than single numbers.

The closest comparisons of learning and classical control are on quadrotors. Hwangbo et al. (2017)
trained a neural policy that stabilises a quadrotor even after it is thrown into the air. Koch et al.
(2019) compared PPO, TRPO and DDPG with a PID for quadrotor attitude control in simulation. Kaufmann et
al. (2023) beat human champions at drone racing with deep RL, and Song et al. (2023) compared optimal
control with RL on the same racing task. This project asks the same question for a flapping body:
unstable in hover, with direction-dependent drag, controlled through wing amplitudes and stroke tilts
rather than four propellers, and with the classical controllers evaluated on exactly the same episodes as
the learned ones.

# 3. The setup

## 3.1 The simulated bird

The bird is one rigid body of 4 g with a 10 cm wingspan. Each wing produces a thrust proportional to the
square of its mean speed:

$$F = \tfrac{1}{2}\,\rho\, C_L\, S\,(2\pi f A r)^2$$

where ρ is the air density, C<sub>L</sub> a lift coefficient, S the wing area, f the wingbeat frequency, A
the amplitude and r the distance from the shoulder to the middle of the wing. The controller therefore
steers by changing the two amplitudes, the frequency and the tilts of the stroke planes. The difference between the wings rolls
the bird, tilting both stroke planes pitches it, tilting them in opposite directions turns it. Three
ingredients make the problem realistic:

- **Hover instability.** A speed-dependent moment tilts the bird further into its own motion, so an uncontrolled bird develops an oscillation that doubles in about 0.75 s, as in hovering insects and helicopters.
- **Direction-dependent drag.** Flying head-first costs half the drag of flying tail-first, which gives the bird a reason to turn rather than back up.
- **Noise everywhere.** Gusts (an Ornstein-Uhlenbeck process), 1% sensor noise and actuator noise are always on.

Physics runs at 240 Hz and the controller decides at 60 Hz. **What we simplify:** the wings are averaged
over each wingbeat (no unsteady aerodynamics), wind is a force on the body rather than a change of the air
the wings move through, and the air shown in the demo is computed for the eye, not fed back into the
flight.

## 3.2 Actions and observations

Every controller sends five numbers in [−1, 1] (left and right amplitudes, frequency, stroke-plane tilt,
opposite tilts) and receives the same 48 noisy numbers: altitude, velocity, roll, pitch, body rates, the
previous action, up to two targets and the five nearest obstacles. Everything is expressed in the bird's
heading frame, so a target 30 cm ahead looks the same whichever way the bird faces, and the layout is the
same for every task, which makes transfer between tasks possible. Wind, mass and wing damage are not
observed.

## 3.3 The four tasks

| Task | Episode | Goal | Success |
|---|---|---|---|
| Hover & recover | 5 s | catch the bird after a strong push (±10 cm, ±15°, ±2 rad/s) and hold the start point | mean distance over the last 2 s below 3 cm |
| Go-to-point | 5 s | fly to a point 10 to 50 cm away in any direction, about half of them behind the bird | hold the target: within 3 cm, slower than 10 cm/s, for 0.25 s |
| Obstacles | 6 s | reach one goal 50 to 80 cm away; 3 to 6 spheres sit on and around the straight line to it, so at least one blocks the way | hold the goal without touching a sphere |
| Navigation | 15 s | a course of three waypoints in a fixed order (25 to 50 cm apart, some behind the bird), with 3 to 6 spheres around the course; the first two only need to be passed within 4 cm, without stopping | reach all three in order, hold the last one, touch nothing |

Rewards are bounded and mostly positive near the goal (otherwise crashing early becomes the best
strategy), with small penalties for jerky actions and energy and −10 for a crash, a loss of control
(tilt beyond 80°) or a collision.

## 3.4 Disturbances

| | Mean wind (% of weight) | Gusts | Mass | Wing damage |
|---|---|---|---|---|
| **Normal** (training and tests) | 0 to 5% | 2% | nominal | none |
| **Randomised** (robustness training) | 0 to 30% | up to 10% | ±25% | up to 40% on one wing, in 70% of the episodes |
| **Robustness grid** (tests only) | 0, 10, 20, 30, 40% | max(2%, wind/3) | nominal | 0, 10, 20, 30, 40% of the left wing's force |

The grid goes beyond the randomised training range (40% wind instead of 30%) to test generalisation, not
only interpolation. Two more tests: a mass sweep from 0.6 to 1.4 times the nominal mass, and a sudden 40%
failure of one wing at a random moment between 1.5 and 3 s, never seen in training.

## 3.5 Sanity check

The constant hover action, replayed without feedback, succeeds in 0% of the hover episodes: the bird loses
control after about 1.2 s. The task really needs a controller.

## 3.6 The interactive demo

The [demo](https://nca1431.github.io/hummingbird-rl/) runs in any browser, with nothing to install. The
simulator is ported line by line to TypeScript and the trained networks are exported, so the bird you see
is flown by the same physics and the same policies as in this report. **Recorded** replays test episodes
exactly as they were measured; **Live** draws new random episodes; in the **Sandbox** you place targets
and obstacles, change the wind, the mass and the damage of each wing, and fly up to four birds with
different controllers side by side. The 3D hummingbird beats its wings with the measured kinematics, and
the air around it is a real-time fluid simulation pushed by the wings, shown as smoke, a laser sheet or
vortex tubes.

# 4. Methods

## 4.1 Controllers

- **PID**: a cascade of position, velocity and attitude loops, as in quadrotor autopilots, with integrals that cancel a steady wind and the torque of a damaged wing. It never turns: it flies sideways or backwards.
- **LQR**: the optimal linear controller for the bird linearised around hover, with an integral of the position error. Its cost-to-go xᵀPx is the exact value function of the linearised problem, which Section 5.7 compares with PPO's critic.
- **Potential field**: for obstacles, both classical controllers follow a target pushed away from nearby spheres.
- **PPO**: on-policy actor-critic with a clipped objective; reliable and cheap per sample.
- **TD3** and **SAC**: off-policy actor-critics that reuse every transition from a replay buffer; SAC adds an entropy bonus.

The [algorithms report](https://github.com/NCA1431/hummingbird-rl/blob/main/docs/algorithms_report.pdf)
explains every component with its equation, the line of code that computes it and the ablation that
shows its effect.

## 4.2 Protocol

- **Budgets**, in environment steps, one step being one decision of the controller: PPO 1 million (hover), 1.5 million (go-to-point), 3 million (obstacles), 4 million (navigation); SAC 0.5, 0.5, 1 and 1 million; TD3 0.5 million on hover and go-to-point. At 60 decisions per second, one million steps is about 4.6 hours of simulated flight. The budgets are about three times what the pilot runs needed to reach 90% validation success.
- **Seeds**: 3 per configuration, 5 for PPO on Obstacles and Navigation and for the transfer experiment. Seed 0 was used for pilots and tuning and is never reported.
- **Validation**: 50 fixed episodes, evaluated regularly during training (the learning curves). They happen to contain a few very hard starts that every model fails, which is why PPO's hover validation success (about 91%) is lower than its test success (98%), and why reported numbers come from fresh test episodes.
- **Tests**: 200 held-out episodes per run, the same for every controller; 20 episodes per cell of the robustness grid, 20 per mass, 100 sudden failures. Reported numbers are from the **final** model; for SAC and TD3 the best-validation checkpoint is shown next to it.
- **Statistics**: means over seeds, with every seed shown; 95% Wilson intervals for success rates; a paired t-test for transfer.
- **Compute**: one desktop PC, seven runs at a time; PPO trains at about 700 steps per second, TD3 86 and SAC 43.

# 5. Results

## 5.1 Benchmark (E1)

**Everyone hovers and reaches points; only learning gets around obstacles.** In calm air the classical
controllers are perfect on the two easy tasks: PID and LQR succeed in 100% of the hover and go-to-point
tests, and the PID is the most precise (0.44 cm on hover). PPO reaches 98% on hover and 100% on
go-to-point, with the best precision on go-to-point (0.65 cm). On Obstacles the picture flips: SAC
succeeds in 98% and PPO in 89% of the episodes (five seeds: 90, 92, 89, 82, 94), against 44% for the PID
and 52% for the LQR, whose potential field gets stuck in front of the spheres. The learned agents also
fly there with a third of the energy (0.47 J against 1.40 J), because they finish much sooner.

<!-- panels: e1_hover -->
<!-- panels: e1_goto -->
<!-- panels: e1_obstacles -->
<!-- panels: e1_navigation -->

**Navigation is the hardest task, and the PID wins it.** PID 41%, LQR 28%, PPO 18%, SAC 10%. The three
families fail in three different ways: the PID runs out of time in front of an obstacle, the LQR loses
control when the field pushes it far from hover (41% of its episodes), and the learned agents collide
(70 to 76% for PPO). Our reward only punishes touching an obstacle, never coming close to one, so nothing
teaches the agents to keep a margin.

**Sample efficiency against wall-clock time.** SAC needs about three times fewer samples than PPO to reach
90% validation success on Obstacles (0.45 to 0.65 million steps against 0.95 to 2.0 million), but on our CPU it trains
about sixteen times slower per step, so PPO finished in 68 minutes where SAC took seven hours. With a simulator
this cheap, PPO is the practical choice; with real flights, SAC's sample efficiency would be decisive.

**Caveat:** the hover numbers of SAC and TD3 are those of their final models, which suffer from the
instability discussed in Section 5.3; their best checkpoints score 99 and 100%.

## 5.2 Flight behaviour (E1b)

**The learned agents turn; the classical controllers never do.** For a target behind the bird, the PID and
the LQR back up or slide sideways in 96 to 98% of the episodes: their yaw is only damped. Every learned
agent turns instead (PPO 100%, SAC 94%, TD3 93% of the episodes with a target behind) and flies head-first
for most of the transit (PPO 62% of the moving time, SAC 57%, against 12% for the PID). They found the
drag asymmetry: head-first flight costs half the drag. They also turn when the target is ahead (90 to
100%), so "turning" is part of their flying style rather than a decision made for targets behind.

<!-- panels: e1b -->

In the episode chart, each dot is one test episode: hover it for its details, click a controller in the
legend to show it alone. A bird that turns to face its target lands on the dashed diagonal.

**Caveat:** SAC's turn follows the target's bearing, as a bird facing its goal would; PPO's and TD3's turns
follow other patterns that we have not yet analysed episode by episode.

## 5.3 Robustness (E2)

**Trained in calm air, every learned policy is fragile.** On the 5 × 5 grid of wind × wing damage, the
robust area (mean success over the grid) is 0.82 for the PID, 0.57 for the LQR, but only 0.39 for PPO and
0.08 to 0.32 for SAC and TD3. Most of their failures are drift: the bird stays airborne but settles a few
centimetres off target under a steady wind. A feed-forward network maps the current observation to an
action; it has no integral of the error, and the wind is not in the observation.

<!-- panels: e2_area -->
<!-- panels: e2_heatmaps -->

Each cell is the success over 20 episodes, averaged over the seeds: hover it for the exact value. On
hover, SAC and TD3 are shown at their best checkpoint, since their final models degrade (see below).

**Randomised training fixes most of it, at almost no cost.** Trained with random wind, mass and damage,
PPO's robust area rises from 0.39 to 0.79 on hover and from 0.36 to 0.96 on go-to-point, better than the
PID (0.81). SAC (0.82) and TD3 (0.93) improve as much on go-to-point. The fragility was mostly about the
training data, not the algorithm. In normal conditions PPO-randomised loses almost nothing (96% instead
of 98% on hover, 0.94 cm instead of 0.78).

<!-- panels: e2_sweeps -->

**A sudden wing failure becomes almost invisible.** When one wing suddenly loses 40% of its force, the PID
always survives but drifts 9.1 cm away before recovering in 1.7 s; the LQR, which trusts its model of an
intact bird, loses control in 96% of the episodes. PPO trained in calm air deviates by 4.1 cm, and PPO
trained in randomised conditions by 1.5 cm: it never even leaves the 3 cm success ball (median).

<!-- panels: e2_failure -->

**The off-policy agents are unstable during training.** On hover, the validation success of SAC and TD3
swings from one evaluation to the next (SAC seed 1: 100%, 48%, 0%, 94%, 74%), while PPO's curve is
smooth. Their final models are therefore a lottery: SAC ends at 64% but its best checkpoint scored 99%,
TD3 seed 3 ends at 7% but scored 100%. A decaying learning rate, which fixed a similar drift of PPO in our
pilots, does not cure it (SAC 76%, TD3 64% at the end), and SAC's learned temperature does not collapse
when the success drops. The practical lesson is to select off-policy agents on validation.

<!-- panels: e2_offpolicy -->

**Caveats:** the robust area averages 25 cells of 20 episodes, and 40% wind is beyond the training range;
restricted to winds up to 30%, PPO-randomised reaches 0.86 on hover and the PID 0.94. Randomised training
is not better everywhere: on hover, PPO-randomised is slightly less successful than PPO-normal at 1.1 to
1.2 times the nominal mass (88% against 93 to 98%), though it holds better at 0.6.

## 5.4 Fine-tuning (E3)

**Adapting a calm-air policy works, given enough steps.** Starting from PPO trained in normal conditions
and continuing in randomised ones, +30% of the original budget gets about two thirds of the way (robust
area 0.67 on hover, 0.74 on go-to-point), and +100% catches up with training from scratch (0.82 and 0.91,
against 0.79 and 0.96). Nothing is forgotten: success in normal conditions stays at 96 to 97% on hover and
100% on go-to-point. The pre-training is not wasted, but it does not save randomised steps either.

<!-- panels: e3 -->

## 5.5 Ablations (E4)

**Some parts of PPO are essential on this bird, others are not.** Each variant changes one component and
keeps everything else (hover, 3 seeds):

- **No clipping: 0%.** Ten epochs on the same batch move the policy too far from the data; on an unstable bird one bad update means a crash, and the next batch is all crashes.
- **GAE λ = 0: 0%.** One-step advantages see the immediate crash penalty but not the delayed reward of coming back: the bird stays airborne, 40 to 80 cm from its target. λ = 1 gives 94%, noisier.
- **No advantage normalisation: 99%.** No effect, and still none when the rewards are not normalised either (96% against 96%): we first thought one normalisation made the other redundant, and the test refuted it. The likelier reason is Adam, whose updates do not depend on the scale of the gradients.
- **No hover start: 98%,** but twice as many steps to reach 90% validation (0.23 million steps instead of 0.12 million).
- **SAC with a fixed temperature (α = 0.2): 85%** instead of 64%. The learned α settles around 0.02, ten times lower, so the fixed one keeps the policy noisier.
- **A harsher failure penalty on Navigation (−50): 10%** instead of 18%. A penalty that only arrives on contact stays sparse, however large.

<!-- panels: e4 -->

## 5.6 Transfer across tasks (E5)

**Starting Navigation from the Obstacles agent helps, but five seeds cannot prove it.** Initialised from
the PPO agent trained on Obstacles, Navigation reaches 27% test success against 18% from scratch: +9
points on average, 4 of 5 seeds better (seed by seed: 19 → 29, 20 → 27, 10 → 33, 20 → 12, 20 → 34). A
paired t-test over the seeds gives p = 0.16: suggestive, not conclusive.

<!-- panels: e5 -->

## 5.7 Interpretability (E6)

**PPO's critic ranks states like the LQR's exact value, and PPO flies like the PID.** For a bird at rest
in the vertical plane through the hover point, PPO's learned value and the LQR's cost-to-go xᵀPx
order the states almost identically within 5 cm (Spearman rank correlation 0.96, averaged over three
seeds). Both judge a horizontal error worse than a vertical one: the bird must tilt before it can move
sideways, but it can climb at once. Further out (±15 cm) the correlation drops to 0.88 and the seeds
disagree with each other: the critic extrapolates beyond the states it saw in training. The closed-loop
flights tell a different story: the LQR corrects its height first and then slides across, whereas PID and
PPO correct both at once. PPO's paths are 1.1 cm from the PID's on average, and 1.7 cm from the LQR's.

<!-- panels: e6 -->

# 6. Discussion

**What worked, and what surprised us.** Every algorithm learned to fly an unstable body from scratch in
minutes to hours of CPU time. What surprised us most is how much the training data matters compared with
the algorithm: the same networks go from fragile to more robust than a PID simply by training them under
random conditions, and a policy trained with damaged wings shrugs off a sudden failure it has never seen.
The second surprise is the instability of the off-policy agents, invisible in their best checkpoints and
obvious in their final models.

**Failure cases we caught.** Three things looked fine in the numbers and were not. A measurement bug
gave every PPO agent a recovery time of 0.01 s after a wing failure (it counted the first moments inside
the 3 cm ball, before the bird had even drifted out); it was replaced by a settling time. Our first
"turned" measure suggested the agents turn for targets behind them, until we saw that they also turn for
targets ahead; we added the net turn and the time spent flying head-first. And watching PPO in the live
viewer showed that almost all of its rare hover failures are instant overshoots: after a strong push it
commands full roll, and the very responsive roll axis tips the bird past 80° within a few decisions.

**When to choose what.** For a bird that hovers and reaches points in conditions you can model, a
well-tuned PID is hard to beat: it is precise, robust thanks to its integrators, and needs no training.
RL pays off when the task needs decisions that are hard to write by hand, such as getting around
obstacles, and when it is trained in the conditions it will meet.

# 7. Limitations

- **The physics is simplified.** The wings are averaged over each wingbeat. Their force always points out of the bird's back: tilting a stroke plane makes a torque, and the small horizontal push it would also give is ignored. Drag is linear in speed. The bird is one rigid body (the wings move only in the drawing), a heavier bird is uniformly denser, and the integration is first-order (semi-implicit Euler at 240 Hz).
- **Wind is a push on the body.** It does not change the airspeed of the wings, and the hover instability uses the speed over the ground, not through the air.
- **Energy is a relative measure:** the mechanical power of the lift only, without the cost of swinging the wings or muscle efficiency.
- **Two hummingbirds.** The wing motion comes from a calliope hummingbird, the physics parameters are closer to a ruby-throated one; only the animation uses the motion.
- **The air in the demo is illustrative.** Its cells are 1 cm wide, far too coarse for the flow around a wing: it shows the jet below the bird, its bending by the wind and the large eddies, and it does not act back on the flight.
- **Simulation only;** nothing here has flown in the real world.
- **Statistics.** Three seeds for most configurations give wide intervals.
- **Unequal tuning.** The PID and the LQR were tuned by hand on validation episodes, PPO in a few pilot runs, SAC and TD3 kept their standard settings.
- **Hidden wind.** The observation hides the wind, so a memoryless policy can be robust to it but cannot adapt to it.

# 8. Future work

**Closing the gaps found here.**

- *A proximity penalty:* a small cost that grows as the bird approaches an obstacle, so that the agents learn to keep a margin instead of colliding (Section 5.1).
- *Memory:* a recurrent network, or the integral of the position error as an input, would let a policy estimate the wind over time and cancel it, as the PID's integrator does (Section 5.3).
- *Stabler off-policy training:* a replay buffer that keeps the hard starts, a slowly averaged copy of the actor for evaluation, larger validation sets for choosing the model (Section 5.3).
- *A stronger classical baseline:* model predictive control, which optimises the next second of flight at every decision and handles obstacles and limits explicitly.

**Richer physics.**

- *Wingbeat-resolved forces,* from the measured lift of each half-stroke, at the price of a simulation step about eight times smaller.
- *Two-way air:* the demo's fluid simulation already moves the air with the wings; letting that air act back on them would make gusts, ground effect and the bird's own wake part of the flight.
- *Articulated wings and contacts in MuJoCo,* as in flybody, which perching or landing would need.
- *More weather,* from the original plan of the project: rain and hail hitting the bird, thinner air at altitude, other gravities.

**Richer behaviour,** also from the original plan.

- *Foraging:* visiting flowers in an order the bird chooses, some of them toxic, so that it must learn which ones to avoid.
- *Takeoff, landing and perching* on a branch.
- *Hunting insects:* intercepting a moving target.
- *Predators and rivals:* escaping, and the territorial chases of several hummingbirds (multi-agent RL).
- *Vision:* seeing the scene through simulated ray casts instead of being given the positions of objects.

**Learning and the real world.**

- *The three algorithms written from scratch,* to check every implementation detail ourselves.
- *Sim-to-real:* transferring a policy to a flapping-wing robot.

# 9. Appendix

**Hyperparameters.** PPO: 8 environments × 512 steps per update, 10 epochs, minibatches of 256, learning
rate 3·10⁻⁴ decreasing linearly to 0, γ = 0.99, λ = 0.95, clip 0.2, no entropy bonus, value coefficient
0.5, gradient norm 0.5, reward normalisation, separate 2 × 256 tanh networks, initial action noise 0.37.
SAC: 1 environment, buffer 10⁶, 10,000 random steps, batch 256, τ = 0.005, γ = 0.99, learning rate 3·10⁻⁴,
learned temperature, 2 × 256 ReLU. TD3: as SAC with learning rate 10⁻³, policy delay 2, target noise 0.2
(clip 0.5), exploration noise 0.1.

**Physics.** Mass 4 g, wingspan 10 cm, wingbeat 40 Hz (32 to 48), maximum amplitude 80°, hover at 64°,
maximum thrust 2.25 times the weight, angular damping time constant 0.4 s, linear drag 0.0075 (forward),
0.015 (backward), 0.023 (sideways) and 0.020 N·s/m (vertical).

**Verification.** The simulator, the tasks, the controllers and the evaluation have 144 Python tests,
including the calibration targets (hover amplitude, maximum thrust, instability doubling time, damping,
terminal speeds). The browser port reproduces the Python simulator on 20 golden trajectories within
4·10⁻¹² (relative), and changing one coefficient by 0.1% makes 18 of them fail, so the test really bites.
The demo's recorded episodes use the evaluation's own seeds, so it shows exactly what was measured. One
numerical limit: with a damaged wing and no control, the bird can spin faster than 100 rad/s, where the
explicit rotation step diverges; this never happens in an episode, which stops at 80° of tilt.

**Runs.** Training runs are in this project under their names (e.g. `ppo_hover_normal_s1`); every
evaluated controller has one run in the group `results`, with its scores as summary metrics, tagged with
the experiments it belongs to.

**How this project was built.** Hummingbirds fascinate me: they are beautiful, their flight is both
graceful and remarkably precise, and they are the only birds that can hover for long, fly backwards and turn
on the spot. I wanted to learn reinforcement learning on a problem I cared about. I defined the tasks, the
experiments and the comparisons, chose between the options at each step, ran every training and evaluation
on my own computer and interpreted the results. I used Claude (Anthropic) to write most of the code (the
simulator, the training and evaluation pipeline, the web demo) and to draft the text of both reports from my
results, which I reviewed, questioned and corrected. I studied the simulator, the tasks, the classical
controllers and the learning algorithms closely, and the training setup, the configurations, the evaluation
suites, the analysis and the tests in less depth. The rest (the export to the browser, the scripts and the
web demo) I checked from the outside: what each part takes in, what it returns and why.

**A first attempt.** This is my second version of the project. During the summer holidays I spent three
weeks on a first hummingbird simulation, also built with AI assistance, set in a much more complex world and
with a large part written in C++: a physics engine in C++, linked to Python for the learning (I got this
bridge between the two languages working), and three 3D viewers, about 15,000 lines of C++ in all. It grew
too large to finish, and reviewing it showed me why. The pieces never came together, some tasks did not test
what they claimed (a 30-line hand-written controller already solved most of one), and the C++ engine was not
the bottleneck: physics took only about 28% of the training time. So I started again, smaller, in Python,
with classical baselines from the start.

**What I learned.** Some flight physics, at the level of a simplified model: how the wingbeat frequency,
the amplitude of each wing and the tilt of the stroke planes produce the lift and the rolling, pitching
and yawing torques that steer the bird, and why hovering is unstable. How a PID and an LQR stabilise an unstable system, and why the LQR is optimal only for
its own model. How PPO, TD3 and SAC work, part by part, and which of their components and hyperparameters
matter: on this bird, clipping and the GAE λ decide between success and failure, advantage normalisation
changes nothing, and the off-policy agents must be chosen on validation. And how to compare learned and
classical controllers fairly: the same held-out episodes, several seeds, intervals rather than single
numbers.

**What I would do differently.** Even in its second version, the project combines flight physics, fluid
dynamics, rigid-body mechanics, control theory, reinforcement learning, statistics, numerical methods and a
large codebase with a 3D web application: too much to master alone at this stage. Starting with a simpler
problem would have been the better way to learn RL. But it taught me how much scoping matters, and that a
good scope needs a real understanding of the problem and of the tools it will use, which I only gained by
trying. My next project is to implement PPO from scratch.

**Credits.** Wing kinematics: Song, Tobalske, Powers, Hedrick and Luo (2016), Dryad dataset
doi:10.5061/dryad.8ch1b. 3D model: "Animated Hovering Flying Hummingbird Loop" by LasquetiSpice
(Sketchfab), CC BY 4.0.

**References.** Agarwal et al. (2021), Deep RL at the edge of the statistical precipice, NeurIPS ·
Hwangbo et al. (2017), Control of a quadrotor with reinforcement learning, IEEE RA-L 2(4) ·
Kaufmann et al. (2023), Champion-level drone racing using deep reinforcement learning, Nature 620 ·
Koch et al. (2019), Reinforcement learning for UAV attitude control, ACM TCPS 3(2) ·
Song, Romero et al. (2023), Reaching the limit in autonomous racing: optimal control versus RL, Science Robotics 8 ·
Fei et al. (2019), Flappy Hummingbird, ICRA · Fujimoto et al. (2018), TD3, ICML · Haarnoja et al. (2018),
Soft Actor-Critic, ICML · Henderson et al. (2018), Deep RL that matters, AAAI · Raffin et al. (2021),
Stable-Baselines3, JMLR · Schulman et al. (2017), PPO, arXiv:1707.06347 · Song et al. (2016), R. Soc.
Open Sci. 3:160230 · Tobin et al. (2017), Domain randomization, IROS · Vaxenburg et al. (2025), Nature 643.
