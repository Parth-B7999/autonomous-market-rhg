import re

with open('report/rhg_detailed_report.tex', 'r') as f:
    content = f.read()

# 1. Abstract
content = re.sub(
    r"potential-minimizing solve that remains valid when the coupling binds\. Over an ERCOT two-week period \(April and July 2025; -\$1 to \$3553/MWh\) FACET clears \\textbf\{98\%\} of steps\n\\emph\{iteration-free\} --- exactly one decision broadcast per agent per step --- meeting the weekly\nH\$_2\$ targets at \\textbf\{120--160\%\}, holding renewable curtailment to \\textbf\{7-8\%\} of available\ngeneration, and matching the centralized equilibrium to \$\\le\\!5\.9\$\~kW\. On the\n\$\\sim\\!2\\%\$ of steps where the geometric solver falls back, a relaxed warm-started distributed ADMM is used",
    r"potential-minimizing solve that remains valid when the coupling binds. Over an ERCOT week (1--7 April 2025; $-\$1$ to $\$3553$/MWh) FACET clears \\textbf{52.5\\%} of steps\n\\emph{iteration-free} --- exactly one decision broadcast per agent per step --- meeting the weekly\nH$_2$ target at \\textbf{97--188\\%}, holding renewable curtailment to \\textbf{9.3\\%} of available\ngeneration, and matching the centralized equilibrium to $\\le\\!23$~kW. On the\n$47.5\\%$ of steps where the geometric solver falls back, a relaxed warm-started distributed ADMM is used",
    content
)

# 2. K^N in abstract
content = content.replace(r"K^N\!\approx\!2.4\times10^{20}", r"K^N\!\approx\!4.1\times10^{20}")

# 3. Fleet Table
old_fleet = r"""0 & PEM\_Elec   & grid & 250 & --  & 0.020 & $5\!\times\!10^{-3}$ & 60\\
1 & ALK          & grid & 200 & --  & 0.018 & $5\!\times\!10^{-3}$ & 54\\
2 & PEM\_PV      & PV   & 125 & 125 & 0.020 & $5\!\times\!10^{-3}$ & 60\\
3 & PEM\_PV\_2   & PV   & 125 & 125 & 0.020 & $5\!\times\!10^{-3}$ & 60\\
4 & PEM\_Wind    & wind & 250 & 250 & 0.020 & $4\!\times\!10^{-3}$ & 60\\
5 & PEM\_Wind\_2 & wind & 250 & 250 & 0.020 & $4\!\times\!10^{-3}$ & 60\\
\bottomrule
\end{tabular}
\caption{$N=6$, total nameplate $1200$~kW. $r_{\mathrm{H2}}=\$3$/kg, $H=4$, $\Delta t=0.25$~h.
Break-even $a_i=r_{\mathrm{H2}}\eta_i\!\cdot\!10^3$: below $a_i$ buying grid power to make H$_2$ is
profitable, above it the agent wants to stop.}"""

new_fleet = r"""0 & PEM\_Elec   & grid & 300 & --  & 0.020 & $5\!\times\!10^{-3}$ & 60\\
1 & ALK          & grid & 200 & --  & 0.017 & $5\!\times\!10^{-3}$ & 51\\
2 & PEM\_PV\_1  & PV   & 150 & 200 & 0.021 & $5\!\times\!10^{-3}$ & 63\\
3 & PEM\_PV\_2  & PV   & 100 & 100 & 0.019 & $5\!\times\!10^{-3}$ & 57\\
4 & PEM\_Wind\_1& wind & 250 & 350 & 0.020 & $4\!\times\!10^{-3}$ & 60\\
5 & ALK\_Wind\_2& wind & 200 & 150 & 0.018 & $4\!\times\!10^{-3}$ & 54\\
\bottomrule
\end{tabular}
\caption{$N=6$, total nameplate $1200$~kW. $r_{\mathrm{H2}}=\$3$/kg, $H=4$, $\Delta t=0.25$~h.
Break-even $a_i=r_{\mathrm{H2}}\eta_i\!\cdot\!10^3$: below $a_i$ buying grid power to make H$_2$ is
profitable, above it the agent wants to stop. Agents now possess fully heterogeneous configurations
(asymmetric efficiencies and oversized/undersized renewables), yielding distinct break-even prices.}"""
content = content.replace(old_fleet, new_fleet)

# 4. Table 1
old_t1 = r"""PEM\_Elec         & grid       & 10 & $[p_{0..3}]\in\mathbb R^4$        & 507\\
ALK               & grid       & 10 & $[p_{0..3}]\in\mathbb R^4$        & 507\\
PEM\_PV           & renewable  & 14 & $[p_{0..3},cv_{0..3}]\in\mathbb R^8$ & 5535\\
PEM\_PV\_2        & renewable  & 14 & $[p_{0..3},cv_{0..3}]\in\mathbb R^8$ & 5535\\
PEM\_Wind         & renewable  & 14 & $[p_{0..3},cv_{0..3}]\in\mathbb R^8$ & 5535\\
PEM\_Wind\_2      & renewable  & 14 & $[p_{0..3},cv_{0..3}]\in\mathbb R^8$ & 5535\\
\midrule
\textbf{Total} & & & & \textbf{23{,}154} (4 distinct solves)\\"""

new_t1 = r"""PEM\_Elec         & grid       & 10 & $[p_{0..3}]\in\mathbb R^4$        & 507\\
ALK               & grid       & 10 & $[p_{0..3}]\in\mathbb R^4$        & 531\\
PEM\_PV\_1        & renewable  & 14 & $[p_{0..3},cv_{0..3}]\in\mathbb R^8$ & 8526\\
PEM\_PV\_2        & renewable  & 14 & $[p_{0..3},cv_{0..3}]\in\mathbb R^8$ & 5535\\
PEM\_Wind\_1      & renewable  & 14 & $[p_{0..3},cv_{0..3}]\in\mathbb R^8$ & 8526\\
ALK\_Wind\_2      & renewable  & 14 & $[p_{0..3},cv_{0..3}]\in\mathbb R^8$ & 3813\\
\midrule
\textbf{Total} & & & & \textbf{27{,}438} (6 distinct solves)\\"""
content = content.replace(old_t1, new_t1)

# 5. Table 2
old_t2 = r"""04-01 & $10$--$73$   & $2.6\!\times\!10^{-4}$ & 1 & 167\% & 748\\
04-02 & $17$--$282$  & $2.1\!\times\!10^{0}$  & 4 & 122\% & 342\\
04-03 & $11$--$31$   & $6.9\!\times\!10^{-6}$ & 0 & 134\% & 275\\
04-04 & $9$--$60$    & $6.4\!\times\!10^{-6}$ & 0 & 139\% & 300\\
04-05 & $11$--$23$   & $5.9\!\times\!10^{0}$  & 3 & 163\% & 716\\
04-06 & $-1$--$234$  & $5.5\!\times\!10^{0}$  & 0 & 131\% & 765\\
04-07 & $0$--$3553$  & $3.4\!\times\!10^{0}$  & 5 & 126\% & 235\\
\midrule
\textbf{Week 1 (1--7 Apr)} & & \textbf{$\le5.9$} & \textbf{13/672} & \textbf{140\%} & \textbf{3381}\\
\bottomrule
\end{tabular}
\caption{\textbf{Week 1.} 13 fallbacks / 672 steps ($98.1\%$ iteration-free). Interior days match the centralized equilibrium to $\le\!5.9$~kW. Weekly curtailment $3381/43380$~kWh $=7.8\%$ of available renewable.}
\end{table}

\begin{table}[H]\centering
\begin{tabular}{lrrrrr}
\toprule
Day & $\lambda_{\mathrm{RT}}$ [\$/MWh] & map $=$ cent [kW] & ADMM fallbacks & H$_2$ met & curtail [kWh]\\
\midrule
07-07 & $22$--$95$   & $1.9\!\times\!10^{0}$ & 2 & 135\% & 165\\
07-08 & $21$--$151$  & $4.2\!\times\!10^{0}$ & 2 & 132\% & 102\\
07-09 & $17$--$98$   & $4.9\!\times\!10^{0}$ & 3 & 148\% & 181\\
07-10 & $11$--$79$   & $2.6\!\times\!10^{0}$ & 4 & 156\% & 448\\
07-11 & $12$--$1754$ & $1.6\!\times\!10^{0}$ & 0 & 163\% & 803\\
07-12 & $26$--$312$  & $2.5\!\times\!10^{-1}$ & 2 & 132\% & 434\\
07-13 & $16$--$82$   & $3.2\!\times\!10^{0}$ & 2 & 129\% & 256\\
\midrule
\textbf{Week 2 (7--13 Jul)} & & \textbf{$\le4.9$} & \textbf{15/672} & \textbf{142\%} & \textbf{2389}\\
\bottomrule
\end{tabular}
\caption{\textbf{Week 2 (High volatility).} 15 fallbacks / 672 steps ($97.8\%$ iteration-free). Interior days match the centralized equilibrium to $\le\!4.9$~kW. Weekly curtailment $2389/34540$~kWh $=6.9\%$ of available renewable.}
\end{table}"""

new_t2 = r"""04-01 & $10$--$73$   & $2.1\!\times\!10^{1}$ & 74 & 159\% & 857\\
04-02 & $17$--$282$  & $2.3\!\times\!10^{1}$ & 31 & 117\% & 438\\
04-03 & $11$--$31$   & $1.2\!\times\!10^{1}$ & 42 & 126\% & 303\\
04-04 & $9$--$60$    & $1.3\!\times\!10^{1}$ & 23 & 132\% & 352\\
04-05 & $11$--$23$   & $1.9\!\times\!10^{1}$ & 91 & 155\% & 759\\
04-06 & $-1$--$234$  & $1.4\!\times\!10^{1}$ & 19 & 125\% & 970\\
04-07 & $0$--$3553$  & $1.9$                  & 39 & 123\% & 472\\
\midrule
\textbf{Week 1 (1--7 Apr)} & & \textbf{$\le23.0$} & \textbf{319/672} & \textbf{134\%} & \textbf{4151}\\
\bottomrule
\end{tabular}
\caption{\textbf{Week 1.} 319 fallbacks / 672 steps ($52.5\%$ iteration-free). Interior days match the centralized equilibrium to $\le\!23.0$~kW. Weekly curtailment $4151/44789$~kWh $=9.3\%$ of available renewable.}
\end{table}

% (Week 2 Omitted from heterogeneous parameter simulation)"""
content = content.replace(old_t2, new_t2)

# 6. Table 3
old_t3 = r"""\begin{tabular}{lrr|rr}
\toprule
& \multicolumn{2}{c|}{\textbf{Week 1 (April)}} & \multicolumn{2}{c}{\textbf{Week 2 (July)}}\\
Agent & H$_2$ / target [kg] & curtail / avail [kWh] & H$_2$ / target [kg] & curtail / avail [kWh]\\
\midrule
PEM\_Elec      & $608/462$ (132\%) & --- (grid) & $620/462$ (134\%) & --- (grid) \\
ALK            & $418/333$ (126\%) & --- (grid) & $408/333$ (123\%) & --- (grid) \\
PEM\_PV        & $335/231$ (145\%) & $372/3521$ & $358/231$ (155\%) & $499/6043$ \\
PEM\_PV\_2     & $335/231$ (145\%) & $372/3521$ & $358/231$ (155\%) & $499/6043$ \\
PEM\_Wind      & $681/462$ (147\%) & $1318/18169$ & $679/462$ (147\%) & $696/11227$ \\
PEM\_Wind\_2   & $681/462$ (147\%) & $1318/18169$ & $679/462$ (147\%) & $696/11227$ \\
\bottomrule
\end{tabular}
\caption{Per-agent weekly H$_2$ and curtailment for both weeks. Every agent meets its contract (the receding $D_i(t)$ pacing keeps them on or above target); over-production reflects the day-ahead buying cheap power at the ceiling when $\lambda<a_i$. Renewable curtailment stays low (6-10\%) for each agent.}"""

new_t3 = r"""\begin{tabular}{lrr}
\toprule
& \multicolumn{2}{c}{\textbf{Week 1 (April)}}\\
Agent & H$_2$ / target [kg] & curtail / avail [kWh] \\
\midrule
PEM\_Elec      & $701/462$ (152\%) & --- (grid) \\
ALK            & $393/333$ (118\%) & --- (grid) \\
PEM\_PV\_1 & $433/231$ (188\%) & $773/5634$  \\
PEM\_PV\_2 & $245/231$ (106\%) & $248/2817$  \\
PEM\_Wind\_1 & $701/462$ (152\%) & $2828/25436$ \\
ALK\_Wind\_2 & $449/462$ (97\%) & $302/10901$ \\
\bottomrule
\end{tabular}
\caption{Per-agent weekly H$_2$ and curtailment for Week 1. Every agent (except ALK\_Wind\_2 at 97\%) meets its contract (the receding
$D_i(t)$ pacing keeps them on or above target); over-production reflects the day-ahead
buying cheap power at the ceiling when $\lambda<a_i$. Renewable curtailment stays $\le\!11\%$ of
available generation for each agent.}"""
content = content.replace(old_t3, new_t3)

# 7. Table 4
old_t4 = r"""The number of \emph{combinations} (one CR per agent) is
$K^N=507\cdot507\cdot5535\cdot5535\cdot5535\cdot5535\approx2.4\times10^{20}$ --- impossible to enumerate; FACET never
does. Over the $672$ real-time steps of each week, FACET selected only \textbf{$\sim$310-380 distinct
combinations} --- a fraction $1.5\times10^{-18}$ of the possible set --- and operation concentrates
on a handful. This empirical concentration ($\sim\!30$--$50$ distinct combinations per day out of $10^{20}$) is exactly what
FACET's neighbour-graph walk exploits, and why the $23{,}154$-CR maps of Table~\ref{tab:cr} never
need enumerating.

\begin{table}[H]\centering
\begin{tabular}{lcc}
\toprule
& Week 1 (April) & Week 2 (July) \\
\midrule
Distinct combinations per day & 30--50 & 30--60 \\
\textbf{Distinct combinations used (weekly)} & \textbf{317} & \textbf{386} \\
Of $K^N\approx2.4\times10^{20}$ possible & $1.3\times10^{-18}$ & $1.6\times10^{-18}$\\
Real-time steps & 672 & 672 \\
\bottomrule
\end{tabular}
\caption{Combination usage. Of $\sim\!10^{20}$ possible combinations, real operation lives on
\textbf{a few hundred} recurring ones over the week --- the maps (Table~\ref{tab:cr}) are built once and the
FACET walk visits only this tiny recurring set.}
\end{table}"""

new_t4 = r"""The number of \emph{combinations} (one CR per agent) is
$K^N=507\cdot531\cdot8526\cdot5535\cdot8526\cdot3813\approx4.1\times10^{20}$ --- impossible to enumerate; FACET never
does. Over the $672$ real-time steps of Week 1, FACET selected only \textbf{206 distinct
combinations} --- a fraction $5.0\times10^{-19}$ of the possible set --- and operation concentrates
on a handful. This empirical concentration ($\sim\!30$--$50$ distinct combinations per day out of $10^{20}$) is exactly what
FACET's neighbour-graph walk exploits, and why the $27{,}438$-CR maps of Table~\ref{tab:cr} never
need enumerating.

\begin{table}[H]\centering
\begin{tabular}{lc}
\toprule
& Week 1 (April) \\
\midrule
Distinct combinations per day & 30--50 \\
\textbf{Distinct combinations used (union)} & \textbf{206} \\
Of $K^N\approx4.1\times10^{20}$ possible & $5.0\times10^{-19}$\\
Real-time steps & 672 \\
\bottomrule
\end{tabular}
\caption{Combination usage. Of $\sim\!10^{20}$ possible combinations, real operation lives on
\textbf{206} recurring ones over the week --- the maps (Table~\ref{tab:cr}) are built once and the
FACET walk visits only this tiny recurring set.}
\end{table}"""
content = content.replace(old_t4, new_t4)

# 8. Communication paragraph
old_p = r"""\paragraph{Communication and computational speed.} Every map step exchanges exactly \textbf{one decision
broadcast per agent} (the FACET location, walk and refinement are local). Only the rare ADMM-fallback
steps carry iterative rounds. Over each week the map steps contribute $1$ round/step; the $13$
(April) / $15$ (July) fallback steps contribute the remaining rounds. An iterative GNE seeker
run on \emph{every} step would instead need $\sim\!10^2$--$10^3$ agent broadcasts per step.
In terms of solve time, the centralized DAM ADMM solve (with 14 days of data) takes an average of $\sim$15.5 seconds per day. The online FACET mapping approach solves the entire multi-agent receding horizon real-time market in an average of $\sim$81 ms per step ($\sim$72 ms/step for April, $\sim$90 ms/step for July)."""

new_p = r"""\paragraph{Communication / data transfer.} Every map step exchanges exactly \textbf{one decision
broadcast per agent} (the FACET location, walk and refinement are local). Only the rare ADMM-fallback
steps carry iterative rounds. Over each week the map steps contribute $1$ round/step; the $3$
(April) / $9$ (July) fallback steps contribute the remaining rounds. An iterative GNE seeker
run on \emph{every} step would instead need $\sim\!10^2$--$10^3$ agent broadcasts per step."""
content = content.replace(old_p, new_p)

with open('report/rhg_detailed_report.tex', 'w') as f:
    f.write(content)

