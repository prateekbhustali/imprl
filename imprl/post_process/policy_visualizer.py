import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Polygon


class PolicyVisualizer:
    def __init__(self, env, agent, oracle=None, oracle_kwargs=None):

        if env.__class__.__name__ == "SingleAgentWrapper":
            self.single_agent = True
        else:
            self.single_agent = False

        self.env = env
        k = env.core.k
        n = env.core.n_components
        self.env_name = f"{k}-out-of-{n}"
        self.oracle = oracle
        self.oracle_kwargs = {} if oracle_kwargs is None else dict(oracle_kwargs)
        self.agent = agent
        self.time_horizon = env.core.time_limit
        self.num_components = env.core.n_components
        self.num_damage_states = env.core.n_damage_states
        self.num_component_actions = env.core.n_comp_actions

    def get_sample_rollout(self, include_oracle_actions=True):
        def _system_belief(obs):
            if self.env.core.time_perception:
                _, belief = obs
            else:
                belief = obs
            # Core env beliefs: (n_components, n_damage_states).
            return belief

        def _select_behavior_action(obs, time_idx, insp_outcomes):
            if self.agent.__class__.__name__ == "InspectRepairHeuristicAgent":
                return self.agent.select_action(
                    time=time_idx,
                    insp_outcomes=insp_outcomes,
                    beliefs=_system_belief(obs),
                )

            if self.single_agent:
                action_id = self.agent.select_action(
                    self.env.system_percept(obs), training=False
                )
                return self.env.joint_action_map[action_id]
            else:  # multi-agent case
                return self.agent.select_action(obs, training=False)

        def _select_oracle_action(obs, time_idx, insp_outcomes):
            if self.oracle.__class__.__name__ == "InspectRepairHeuristicAgent":
                return self.oracle.select_action(
                    time=time_idx,
                    insp_outcomes=insp_outcomes,
                    beliefs=_system_belief(obs),
                )

            oracle_kwargs = {"training": False, **self.oracle_kwargs}
            return self.oracle.select_action(obs, **oracle_kwargs)

        # data to be collected
        data = {
            "time": np.arange(0, self.time_horizon + 1),
            "true_states": np.empty(
                (self.time_horizon + 1, self.num_components), dtype=int
            ),
            "beliefs": np.empty(
                (self.time_horizon + 1, self.num_components, self.num_damage_states)
            ),
            "actions": np.ones((self.time_horizon, self.num_components), dtype=int)
            * -1,
            "oracle_actions": np.ones(
                (self.time_horizon, self.num_components), dtype=int
            )
            * -1,
            "system_pf": np.zeros(self.time_horizon + 1),
            "rewards": np.zeros(self.time_horizon),
            "cost_mobilisation": np.zeros(self.time_horizon),
            "cost_risk": np.zeros(self.time_horizon),
            "cost_inspections": np.zeros(self.time_horizon),
            "cost_replacements": np.zeros(self.time_horizon),
            "failure_timepoints": np.zeros(self.time_horizon + 1),
            "episode_cost": 0,
        }

        terminated, truncated = False, False
        episode_reward = 0
        time = 0

        observation, info = self.env.core.reset()
        inspection_outcomes = info["inspection_outcomes"]
        system_belief = _system_belief(observation)
        data["beliefs"][0, :, :] = system_belief

        while not truncated and not terminated:
            # damage states
            data["true_states"][time, :] = self.env.core.damage_state

            # systems failure risk
            pf = system_belief[:, -1]
            data["system_pf"][time] = self.env.core.pf_sys(pf, self.env.core.k)

            # select action
            action = _select_behavior_action(observation, time, inspection_outcomes)

            next_observation, reward, terminated, truncated, info = self.env.core.step(
                action
            )

            if include_oracle_actions and self.oracle is not None:
                oracle_action = _select_oracle_action(
                    observation, time, inspection_outcomes
                )
                data["oracle_actions"][time, :] = oracle_action

            # update belief
            system_belief = _system_belief(next_observation)
            data["beliefs"][time + 1, :, :] = system_belief
            data["actions"][time, :] = action

            _discount = self.env.core.discount_factor**time
            data["rewards"][time] = reward * _discount
            data["cost_mobilisation"][time] = info["reward_mobilisation"] * _discount
            data["cost_risk"][time] = info["reward_system"] * _discount
            data["cost_inspections"][time] = info["reward_inspection"] * _discount
            data["cost_replacements"][time] = info["reward_replacement"] * _discount

            # note system failure timepoints
            if info["system_failure"]:
                data["failure_timepoints"][time] = 1

            # update observation
            observation = next_observation
            inspection_outcomes = info["inspection_outcomes"]

            # update episode reward
            episode_reward += reward * _discount

            # update time
            time += 1

        data["episode_cost"] = -episode_reward
        data["true_states"][time, :] = self.env.core.damage_state
        data["system_pf"][time] = self.env.core.pf_sys(
            system_belief[:, -1], self.env.core.k
        )

        return data

    def get_sample_rollout_batch(
        self, num_trajectories=1, include_oracle_actions=False
    ):
        if num_trajectories < 1:
            raise ValueError("num_trajectories must be at least 1.")

        rollouts = [
            self.get_sample_rollout(include_oracle_actions=include_oracle_actions)
            for _ in range(num_trajectories)
        ]
        return {
            key: np.asarray([rollout[key] for rollout in rollouts])
            for key in rollouts[0]
        }

    def _setup_plot(self):

        # Use the custom notebook-style layout for 2 and 4 components
        if self.num_components == 2:
            mosaic = """
                111222
                ..BB..
            """
            figsize = (15, 5)
        elif self.num_components == 3:
            # Two components on top. Bottom: Component 3 wide, bar shifted left,
            # leaving extra blank space on the far right for the legend.
            mosaic = """
                111222
                ..333.
                ..BB..
            """
            figsize = (15, 7)
        elif self.num_components == 4:
            mosaic = """
                111333
                222444
                ..BB..
            """
            figsize = (15, 7)
        elif self.num_components == 5:
            # Five-component layout (two on top, two middle, one bottom-right; bar bottom-left)
            mosaic = """
                1155
                2244
                B33.
            """
            figsize = (15, 7)
        else:
            # Fallback
            mosaic = """
                12
                B.
            """
            figsize = (12, 5)

        fig = plt.figure(figsize=figsize)
        ax_dict = fig.subplot_mosaic(mosaic)

        self.time_horizon_ticks = np.arange(0, self.time_horizon + 1, 2)
        self.action_markersize = 8

        plt.rcParams.update(
            {
                "axes.titlesize": "medium",
                "axes.labelsize": "large",
            }
        )

        self.action_idxs = np.arange(self.num_component_actions)
        self.action_labels = ["do-nothing", "repair", "inspect"]
        self.action_colors = ["gray", "darkviolet", "orange"]
        self.action_markers = [".", "s", ">"]

        for c in range(self.num_components):
            ax = ax_dict[f"{c+1}"]

            ## Plot agent actions
            ax2 = ax.twinx()
            ax2.set_yticks(self.action_idxs)

            ax.set_ylim([-0.05, 1.05])
            ax.set_xlim([-0.5, self.time_horizon + 0.5])
            ax.set_yticks([0, 0.5, 1])
            ax.set_xlabel("time", fontsize=15)
            ax.set_title(f"Component {c+1}", weight="bold", fontsize=16, pad=15)
            ax.grid()

            ax2.tick_params(right=False, labelright=False)
            ax2.spines[["top", "right", "left"]].set_visible(False)

        # create legend handles
        legend_handles = []
        for a in self.action_idxs:
            legend_handles += [
                Line2D(
                    [],
                    [],
                    marker=self.action_markers[a],
                    markersize=self.action_markersize,
                    label=self.action_labels[a],
                    color=self.action_colors[a],
                    linestyle="None",
                )
            ]

        labels = ["mobilization", "repair", "inspect", "risk"]
        colors = ["lightpink", "darkviolet", "orange", "lightcoral"]

        barplot = ax_dict["B"].barh(labels, [0] * len(labels), color=colors, height=0.4)
        ax_dict["B"].set_xlim([0, 100])
        ax_dict["B"].set_xticks([0, 25, 50, 75, 100])
        ax_dict["B"].set_xticklabels(["0%", "25%", "50%", "75%", "100%"])

        return fig, ax_dict, legend_handles, barplot

    def plot(self, data=None, save_fig_kwargs=None):

        if data is not None:
            self.data = data
        else:
            # get data from agent rollout
            self.data = self.get_sample_rollout()

        # get base plot from Plotter
        fig, ax_dict, legend_handles, barplot = self._setup_plot()

        _y_action = 1

        for c in range(self.num_components):
            ax = ax_dict[f"{c+1}"]

            # belief
            # beliefs layout: (time, component, damage_state)
            comp_belief = self.data["beliefs"][:, c, :]
            colorbar = ax.pcolormesh(
                self.data["time"],
                np.arange(self.num_damage_states),
                comp_belief.T,
                shading="nearest",
                cmap="binary",
                alpha=0.2,
                vmin=0,
                vmax=1,
                edgecolors="face",
            )

            # true state
            (h_true_state,) = ax.plot(
                self.data["time"],
                self.data["true_states"][:, c],
                "-",
                label="damage state",
                color="tab:blue",
                markersize=2,
                alpha=0.8,
            )

            # system_pf
            ax2 = ax.twinx()
            _risk = self.data["system_pf"]
            max_risk = max(_risk)
            (h_system_pf,) = ax2.plot(
                self.data["time"],
                _risk / max_risk,
                "-",
                label=r"system $p_f$",
                color="tab:orange",
                markersize=2,
                alpha=0.3,
            )

            ax2.set_ylim([0, 1.5])
            ax2.set_yticks([0, 1, 1.5])
            ax2.set_yticklabels(
                [0, f"{max_risk:.2f}", ""], color="tab:orange"
            )  # Rescale to [0, 1]
            ax2.set_ylabel(
                r"system $p_f$", fontsize=12, color="tab:orange", loc="bottom"
            )

            ax.set_yticks(np.arange(self.num_damage_states))
            if self.oracle is not None:
                ax.set_ylim([-0.5, self.num_damage_states + 0.48])
            else:
                ax.set_ylim([-0.5, self.num_damage_states - 0.48])
            ax.set_xticks(self.time_horizon_ticks)
            ax.set_xticklabels(self.time_horizon_ticks, fontsize=14)

            # draw vertical lines when system fails
            if self.data["failure_timepoints"].sum() > 0:
                ax.vlines(
                    np.where(self.data["failure_timepoints"]),
                    -1,
                    self.num_damage_states - 0.48,
                    label="system-failure",
                    color="red",
                    alpha=0.5,
                )

            for a in self.action_idxs:
                # Agent actions
                _x = np.where(self.data["actions"][:, c] == a)
                ax.plot(
                    _x,
                    _y_action,
                    self.action_markers[a],
                    markersize=self.action_markersize,
                    label=self.action_labels[a],
                    color=self.action_colors[a],
                )

                # Oracle actions
                if self.oracle is not None:
                    # shade oracle actions region
                    ax.fill_between(
                        np.arange(-1, self.time_horizon + 2),
                        self.num_damage_states - 0.48,
                        self.num_damage_states + 0.48,
                        color="green",
                        alpha=0.05,
                    )
                    _x = np.where(self.data["oracle_actions"][:, c] == a)
                    ax.plot(
                        _x,
                        self.num_damage_states,
                        self.action_markers[a],
                        markersize=self.action_markersize,
                        label=self.action_labels[a],
                        color=self.action_colors[a],
                    )
            ax.set_ylabel("damage-state", fontsize=10, color="tab:blue", loc="bottom")
            if self.oracle is not None:
                ax.spines.top.set_position(("data", self.num_damage_states - 0.48))
            ax.grid(False)

        # update legend_handles
        legend_handles += [h_true_state, h_system_pf]
        if self.data["failure_timepoints"].sum() > 0:
            legend_handles += [
                Line2D([], [], color="red", label="system-failure", alpha=1)
            ]
        fig.legend(handles=legend_handles, loc=(0.83, 0.05), fontsize=12)

        ## Bar plot
        total_costs = np.array(
            [
                -self.data["cost_mobilisation"].sum() + 1e-15,  # numerical stability
                -self.data["cost_replacements"].sum(),
                -self.data["cost_inspections"].sum() + 1e-15,  # numerical stability
                -self.data["cost_risk"].sum(),
            ]
        )

        # Prefer component-derived costs; fall back to episode_cost if consistent
        episode_cost_sum = float(total_costs.sum())
        target_episode_cost = (
            float(self.data["episode_cost"])
            if "episode_cost" in self.data
            else episode_cost_sum
        )
        if not np.isclose(episode_cost_sum, target_episode_cost, rtol=1e-3, atol=1e-6):
            used_episode_cost = episode_cost_sum
        else:
            used_episode_cost = target_episode_cost

        denom = used_episode_cost if abs(used_episode_cost) > 1e-12 else 1.0
        _percentages = total_costs * 100 / denom

        # update bar widths and draw value labels OUTSIDE the bar axis at the right
        for b, bar in enumerate(barplot):
            bar.set_width(_percentages[b])
            y = bar.get_y() + bar.get_height() / 2
            # place just beyond 100% so labels sit outside; allow drawing outside axes
            ax_dict["B"].text(
                102.0,
                y,
                f"{total_costs[b]:.1f}",
                va="center",
                ha="left",
                fontsize=12,
                color="black",
                clip_on=False,
            )

        ax_dict["B"].set_title(f"Episode cost: {used_episode_cost:.3f}", fontsize=14)

        # title
        fig.suptitle(f"{self.env_name} system", fontsize=18, weight="bold")

        # Extract k and n from env_name format "k-out-of-n"
        k_val = int(self.env_name.split("-")[0])
        n_val = int(self.env_name.split("-")[-1])
        if k_val == 1:
            fig.text(
                0.5,
                0.92,
                "(parallel configuration)",
                fontsize=14,
                ha="center",
                va="center",
            )
        elif k_val == n_val:
            fig.text(
                0.5,
                0.92,
                "(series configuration)",
                fontsize=14,
                ha="center",
                va="center",
            )
        fig.tight_layout()

        # colorbar for belief (panel-relative next to bar plot, as before)
        bpos = ax_dict["B"].get_position(fig)
        pad = 0.05  # gap to the right of bar plot (figure-normalized)
        width = 0.022  # colorbar width (figure-normalized)
        cbar_ax = fig.add_axes([bpos.x1 + pad, bpos.y0, width, bpos.height])
        cbar = fig.colorbar(colorbar, cax=cbar_ax)
        cbar.set_label("belief", fontsize=14, weight="bold")
        cbar.ax.tick_params(labelsize=12)
        try:
            cbar.set_ticks([0.0, 0.5, 1.0])
        except Exception:
            pass
        try:
            cbar.outline.set_linewidth(1.0)
        except Exception:
            pass

        # add text block like the reference layout (absolute positions)
        policy_b = self.agent.name
        fig.text(0.05, 0.20, f"Behavior policy: {policy_b}", fontsize=14, weight="bold")
        if self.oracle is not None:
            policy_o = (
                getattr(self.oracle, "name", None)
                or self.oracle.__class__.__name__
            )
            fig.text(
                0.05,
                0.15,
                f"Baseline policy: {policy_o}",
                fontsize=14,
                color="green",
                weight="bold",
            )
            fig.text(
                0.05,
                0.12,
                "(baseline policy actions on top with green background)",
                fontsize=10,
                color="green",
            )

        plt.show()

        if save_fig_kwargs is not None:
            fig.savefig(**save_fig_kwargs)

    def plot_belief_space(
        self,
        show_actions=True,
        data=None,
        num_trajectories=1,
        save_fig_kwargs=None,
        fig=None,
        axs=None,
        show=True,
        trajectory_color="tab:blue",
        initial_belief_color="blue",
    ):
        # Belief-space projection is defined for 3 damage states only.
        if self.num_damage_states != 3:
            raise ValueError(
                "plot_belief_space currently supports exactly 3 damage states."
            )

        cartesian = np.array([[0.0, 0.0], [1.0, 0.0], [0.5, np.sqrt(3) / 2]])

        if data is None:
            data = self.get_sample_rollout_batch(
                num_trajectories=num_trajectories,
                include_oracle_actions=False,
            )
        beliefs = np.asarray(data["beliefs"])
        actions = np.asarray(data["actions"]) if show_actions else None

        tri_beliefs = beliefs @ cartesian
        n_episodes, n_time, n_components, _ = tri_beliefs.shape

        owns_axes = axs is None
        if owns_axes:
            fig, axs = plt.subplots(1, n_components, figsize=(4 * n_components, 3.5))
        elif fig is None:
            fig = plt.gcf()
        axs = np.atleast_1d(axs)
        if len(axs) != n_components:
            raise ValueError(
                f"Expected {n_components} axes (one per component), got {len(axs)}."
            )

        action_labels = ["do-nothing", "repair", "inspect"]
        action_colors = ["gray", "darkviolet", "orange"]
        action_markers = [".", "s", ">"]

        for c in range(n_components):
            ax = axs[c]

            if show_actions:
                # Batch scatter by action type to avoid one draw call per point.
                t_max = min(actions.shape[1], n_time - 1)
                if t_max > 0:
                    action_slice = actions[:, :t_max, c].astype(int).reshape(-1)
                    xs = tri_beliefs[:, :t_max, c, 0].reshape(-1)
                    ys = tri_beliefs[:, :t_max, c, 1].reshape(-1)
                    valid = (action_slice >= 0) & (action_slice < len(action_markers))

                    for a in range(len(action_markers)):
                        mask = valid & (action_slice == a)
                        if not np.any(mask):
                            continue
                        ax.scatter(
                            xs[mask],
                            ys[mask],
                            color=action_colors[a],
                            marker=action_markers[a],
                            facecolor="none",
                            s=28,
                            alpha=0.7,
                        )
            else:
                for ep in range(n_episodes):
                    xs = tri_beliefs[ep, :, c, 0]
                    ys = tri_beliefs[ep, :, c, 1]
                    ax.plot(
                        xs, ys, "-", color=trajectory_color, alpha=0.15, linewidth=1.0
                    )
                    ax.scatter(xs, ys, color=trajectory_color, s=8, alpha=0.2)

            # mark initial belief once per component
            x0, y0 = tri_beliefs[0, 0, c, 0], tri_beliefs[0, 0, c, 1]
            ax.scatter(x0, y0, color=initial_belief_color, s=50, label="initial belief")

            triangle = Polygon(
                cartesian,
                closed=True,
                edgecolor="k",
                fill=None,
                alpha=0.5,
                linewidth=2,
            )
            ax.add_patch(triangle)
            ax.set_xlim(-0.25, 1.25)
            ax.set_ylim(-0.25, np.sqrt(3) / 2 + 0.25)
            ax.text(0.0, -0.1, "no-damage", fontsize=9, ha="center", va="center")
            ax.text(1.0, -0.1, "major-damage", fontsize=9, ha="center", va="center")
            ax.text(
                0.5,
                np.sqrt(3) / 2 + 0.1,
                "failure",
                fontsize=9,
                ha="center",
                va="center",
            )
            ax.set_title(f"Component {c+1}", fontsize=14, fontweight="bold")
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)

        legend_handles = []
        if show_actions:
            for a, lbl in enumerate(action_labels):
                legend_handles.append(
                    Line2D(
                        [],
                        [],
                        marker=action_markers[a],
                        markersize=10,
                        markerfacecolor="none",
                        label=lbl,
                        color=action_colors[a],
                        linestyle="None",
                    )
                )
        else:
            legend_handles.append(
                Line2D(
                    [],
                    [],
                    marker="o",
                    markersize=7,
                    label="belief trajectory",
                    color=trajectory_color,
                    linestyle="-",
                    alpha=0.7,
                )
            )
        legend_handles.append(
            Line2D(
                [],
                [],
                marker="o",
                color=initial_belief_color,
                linestyle="None",
                label="initial belief",
            )
        )

        if owns_axes:
            fig.legend(handles=legend_handles, loc="upper left", ncol=4, fontsize=10)
            fig.suptitle(
                f"Belief Space: {self.env_name} system ({self.agent.name})",
                fontsize=16,
                fontweight="bold",
                y=1.02,
            )
            fig.tight_layout()

        if save_fig_kwargs is not None:
            fig.savefig(**save_fig_kwargs)
        if show:
            plt.show()
        return fig, axs
