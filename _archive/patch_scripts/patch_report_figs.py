with open("simple_game/report_figs.py", "r") as f:
    content = f.read()

# Replace Fig 5 section
old_fig5 = """# ── FIG 5: Prices (DAM & RTM) vs Forecasts ────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
wk = "April 1-7"; DAYIDX = 5; r = data[wk][DAYIDX]

axes[0].plot(np.arange(24), r["lam_da"], 'k-', label="Actual DAM Price", lw=1.5)
axes[0].plot(np.arange(24), r["lam_da_fc"], 'r--', label="Forecast DAM Price", lw=1.5)
axes[0].set_title(f"DAM Prices on {WEEKS[wk][DAYIDX]}")
axes[0].set_xlabel("Hour of day")
axes[0].set_ylabel("Price [$/MWh]")
axes[0].legend()

t_rtm = np.arange(96) * 0.25
axes[1].plot(t_rtm, r["lam_rt"], 'k-', label="Actual RTM Price", lw=1.5)
axes[1].plot(t_rtm, r["lam_rt_fc"], 'b--', label="Nowcast RTM Price (Step 0)", lw=1.5)
axes[1].set_title(f"RTM Prices on {WEEKS[wk][DAYIDX]}")
axes[1].set_xlabel("Hour of day")
axes[1].set_ylabel("Price [$/MWh]")
axes[1].legend()

fig.tight_layout(); fig.savefig(OUT/"fig5_prices.pdf"); fig.savefig(OUT/"fig5_prices.png"); plt.close(fig)"""

new_fig5 = """# ── FIG 5: Prices (DAM & RTM) vs Forecasts ────────────────────────────────────
fig = plt.figure(figsize=(14, 10))
wk = "April 1-7"; DAYIDX = 5; r = data[wk][DAYIDX]

ax1 = plt.subplot(2, 2, 1)
ax1.plot(np.arange(24), r["lam_da"], 'k-', label="Actual DAM Price", lw=1.5)
ax1.plot(np.arange(24), r["lam_da_fc"], 'r--', label="Forecast DAM Price", lw=1.5)
ax1.set_title(f"DAM Prices on {WEEKS[wk][DAYIDX]}")
ax1.set_xlabel("Hour of day")
ax1.set_ylabel("Price [$/MWh]")
ax1.legend()

ax2 = plt.subplot(2, 2, 2)
t_rtm = np.arange(96) * 0.25
ax2.plot(t_rtm, r["lam_rt"], 'k-', label="Actual RTM Price", lw=1.5)
ax2.plot(t_rtm, r["lam_rt_fc"], 'b--', label="Nowcast RTM Price (Step 0)", lw=1.5)
ax2.set_title(f"RTM Prices on {WEEKS[wk][DAYIDX]}")
ax2.set_xlabel("Hour of day")
ax2.set_ylabel("Price [$/MWh]")
ax2.legend()

ax3 = plt.subplot(2, 1, 2)
lam_da_week = np.concatenate([data[wk][d]["lam_da"] for d in range(7)])
lam_da_fc_week = np.concatenate([data[wk][d]["lam_da_fc"] for d in range(7)])
t_week = np.arange(7 * 24)
ax3.plot(t_week, lam_da_week, 'k-', label="Actual DAM Price", lw=1.5)
ax3.plot(t_week, lam_da_fc_week, 'r--', label="Forecast DAM Price", lw=1.5)
ax3.set_title(f"DAM Prices for {wk}")
ax3.set_xlabel("Hour of week")
ax3.set_ylabel("Price [$/MWh]")
ax3.legend()

fig.tight_layout(); fig.savefig(OUT/"fig5_prices.pdf"); fig.savefig(OUT/"fig5_prices.png"); plt.close(fig)"""

content = content.replace(old_fig5, new_fig5)
with open("simple_game/report_figs.py", "w") as f:
    f.write(content)
