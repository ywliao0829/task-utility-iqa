import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.gridspec import GridSpec

try:
    from adjustText import adjust_text
except Exception:
    adjust_text = None

FONT_SCALE = 1.9  # FONT_SCALE_PATCH_190_APPLIED

INPUT_XLSX = 'cross_model_compare300_results.xlsx'
OUTDIR = 'paper_figs_rebuilt'

BASELINE = pd.DataFrame({
    'Method': ['Q-SiT', 'Q-Insight', 'DeQA-Score-Mix3', 'Cognition-Consistency', 'Ours-TCUS'],
    'AUROC': [0.510, 0.502, 0.500, 0.623, 0.925],
    'AP': [0.632, 0.629, 0.649, 0.694, 0.948],
    'Spearman': [0.017, 0.003, 0.001, 0.352, 0.710],
})

METHOD_COLORS = {
    'Q-SiT': '#7aa0d2',
    'Q-Insight': '#e8b06a',
    'DeQA-Score-Mix3': '#b0b0b0',
    'Cognition-Consistency': '#9a84c5',
    'Ours-TCUS': '#cf4e4e',
}

PROVIDER_COLORS = {
    'qwen': '#4C78A8',
    'gpt': '#F28E2B',
    'gemini': '#59A14F',
    'claude': '#9C6ADE',
    'grok': '#7F7F7F',
}

METRIC_ORDER = ['PSNR', 'SSIM', 'Sharpness', 'Brightness-quality', 'Cognition-Consistency', 'TCUS-compare']
METRIC_SHOW = ['PSNR', 'SSIM', 'Sharpness', 'Brightness', 'Cog-Cons', 'Ours-TCUS']
METRIC_RENAME = dict(zip(METRIC_ORDER, METRIC_SHOW))

MODEL_NAME_MAP = {
    'claude-sonnet-4-5-20250929-thinking': 'Claude Sonnet 4.5 (T)',
    'claude-sonnet-4-5-20250929': 'Claude Sonnet 4.5',
    'claude-opus-4-7': 'Claude Opus 4.7',
    'claude-haiku-4-5-20251001': 'Claude Haiku 4.5',
    'claude-haiku-4-5-20251001-thinking': 'Claude Haiku 4.5 (T)',
    'claude-opus-4-5-20251101': 'Claude Opus 4.5',
    'claude-opus-4-5-20251101-thinking': 'Claude Opus 4.5 (T)',
    'claude-sonnet-4-6': 'Claude Sonnet 4.6',
    'claude-sonnet-4-6-thinking': 'Claude Sonnet 4.6 (T)',
    'claude-opus-4-6': 'Claude Opus 4.6',
    'claude-opus-4-6-thinking': 'Claude Opus 4.6 (T)',
    'gemini-2.5-flash-thinking': 'Gemini 2.5 Flash (T)',
    'gemini-2.5-flash-lite': 'Gemini 2.5 Flash-Lite',
    'gpt-5.4-mini': 'GPT-5.4 Mini',
    'gpt-5.1': 'GPT-5.1',
    'gpt-5': 'GPT-5',
    'gpt-5.4': 'GPT-5.4',
    'gpt-5.2': 'GPT-5.2',
    'gpt-5.5': 'GPT-5.5',
    'Qwen2.5-VL-3B': 'Qwen2.5-VL-3B',
    'Qwen2.5-VL-7B': 'Qwen2.5-VL-7B',
    'grok-4-1-fast-reasoning': 'Grok 4.1 Fast (R)',
    'grok-4-1-fast-non-reasoning': 'Grok 4.1 Fast',
    'grok-3': 'Grok-3',
}

RADAR_METHODS = ['Q-SiT', 'Q-Insight', 'DeQA-Score-Mix3', 'Ours-TCUS']
RADAR_AXES = ['AUROC', 'AP', 'Spearman', 'Avg.']
RADAR_COLORS = {m: METHOD_COLORS[m] for m in RADAR_METHODS}


def short_name(x):
    return MODEL_NAME_MAP.get(str(x), str(x))


def style_setup():
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['font.serif'] = ['Times New Roman']
    plt.rcParams['mathtext.fontset'] = 'stix'
    plt.rcParams['pdf.fonttype'] = 42
    plt.rcParams['ps.fonttype'] = 42
    plt.rcParams['font.serif'] = ['Times New Roman']
    plt.rcParams['mathtext.fontset'] = 'stix'
    plt.rcParams['pdf.fonttype'] = 42
    plt.rcParams['ps.fonttype'] = 42
    plt.rcParams['font.serif'] = ['Times New Roman']
    plt.rcParams['mathtext.fontset'] = 'stix'
    plt.rcParams['axes.unicode_minus'] = False
    plt.rcParams['figure.dpi'] = 220
    plt.rcParams['pdf.fonttype'] = 42
    plt.rcParams['ps.fonttype'] = 42
    sns.set_style('whitegrid')


def load_long_summary(path):
    df = pd.read_excel(path, sheet_name='Long Summary')
    rename_map = {}
    if 'Spearman r' not in df.columns and 'Spearman' in df.columns:
        rename_map['Spearman'] = 'Spearman r'
    if 'Task success ratio' not in df.columns and 'Task success rate' in df.columns:
        rename_map['Task success rate'] = 'Task success ratio'
    if rename_map:
        df = df.rename(columns=rename_map)
    df['ModelShort'] = df['Model'].apply(short_name)
    df['MetricShow'] = df['Metric'].map(METRIC_RENAME)
    return df


def make_radar_norm():
    # Return raw values on the common [0, 1] metric scale.
    # Do NOT normalize each axis by its maximum; otherwise Ours-TCUS becomes 1.0 on every axis
    # and the radar chart can be visually misleading.
    look = BASELINE.set_index('Method')
    raw = []
    for m in RADAR_METHODS:
        vals = [
            float(look.loc[m, 'AUROC']),
            float(look.loc[m, 'AP']),
            float(look.loc[m, 'Spearman']),
        ]
        vals.append(float(np.mean(vals)))
        raw.append(vals)
    return np.array(raw, dtype=float)


def make_baseline_heatmap(outdir):
    heat = BASELINE.set_index('Method')[['AUROC', 'AP', 'Spearman']]
    fig, ax = plt.subplots(figsize=(10.0, 5.6))
    sns.heatmap(
        heat,
        ax=ax,
        cmap='RdYlBu_r',
        annot=True,
        fmt='.3f',
        linewidths=1.0,
        cbar=False,
        annot_kws={'fontsize': 22.8, 'fontweight': 'bold'}
    )
    ax.set_title('Baseline Task-Utility Metric Matrix', fontsize=18.0, fontweight='bold', pad=14)
    ax.set_xlabel('')
    ax.set_ylabel('Method', fontsize=14.0)
    ax.tick_params(axis='x', labelrotation=0, labelsize=12.0)
    ax.tick_params(axis='y', labelrotation=0, labelsize=12.0)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, 'baseline_metric_matrix_no_valid_ratio.png'), bbox_inches='tight')
    fig.savefig(os.path.join(outdir, 'baseline_metric_matrix_no_valid_ratio.pdf'), bbox_inches='tight')
    plt.close(fig)


def style_heatmap_xticklabels(ax, fontsize=10.0):
    for label in ax.get_xticklabels():
        label.set_rotation(25)
        label.set_ha('right')
        label.set_rotation_mode('anchor')
        label.set_fontsize(fontsize)


def make_composite(long_df, outdir):
    ap_pivot = long_df.pivot_table(index='ModelShort', columns='Metric', values='Average Precision', aggfunc='first')
    top12 = (
        long_df[long_df['Metric'].eq('TCUS-compare')][['ModelShort', 'Average Precision']]
        .sort_values('Average Precision', ascending=False)
        .head(12)['ModelShort']
        .tolist()
    )
    heat_df = ap_pivot.loc[top12, METRIC_ORDER]
    heat_df.columns = METRIC_SHOW

    violin_df = long_df.copy()

    ap_wide = long_df.pivot_table(
        index=['Provider', 'Model', 'ModelShort'],
        columns='Metric',
        values='Average Precision',
        aggfunc='first'
    ).reset_index()
    baseline_cols = ['PSNR', 'SSIM', 'Sharpness', 'Brightness-quality', 'Cognition-Consistency']
    ap_wide['BestBaseline'] = ap_wide[baseline_cols].max(axis=1)
    ap_wide['Delta'] = ap_wide['TCUS-compare'] - ap_wide['BestBaseline']
    ap_wide = ap_wide.sort_values('Delta').reset_index(drop=True)
    ap_wide[['Provider', 'Model', 'ModelShort', 'BestBaseline', 'TCUS-compare', 'Delta']].to_csv(
        os.path.join(outdir, 'per_model_ap_gain.csv'), index=False
    )

    scatter_df = long_df[long_df['Metric'].eq('TCUS-compare')][
        ['Provider', 'ModelShort', 'AUROC', 'Average Precision', 'Task success ratio']
    ].copy()

    fig = plt.figure(figsize=(26.0, 16.5))
    gs = GridSpec(
        3, 4,
        figure=fig,
        width_ratios=[1.0, 1.0, 1.0, 1.30],
        height_ratios=[1.05, 1.18, 1.34],
        hspace=0.56,
        wspace=0.44,
    )

    # (a)
    methods = ['Q-SiT', 'Q-Insight', 'DeQA-Score-Mix3', 'Cognition-Consistency', 'Ours-TCUS']
    labels = methods[:]
    colors = [METHOD_COLORS[m] for m in methods]
    specs = [('AUROC', 'AUROC'), ('AP', 'Average Precision'), ('Spearman', 'Spearman $r$')]
    top_axes = [fig.add_subplot(gs[0, i]) for i in range(3)]

    for ax, (col, title) in zip(top_axes, specs):
        x = np.arange(len(methods))
        vals = [float(BASELINE.loc[BASELINE['Method'].eq(m), col].iloc[0]) for m in methods]
        bars = ax.bar(x, vals, color=colors, edgecolor='#444444', linewidth=0.8, width=0.70)
        ax.set_ylim(0, 1.02)
        ax.set_xlim(-0.72, len(x) - 0.18)
        ax.set_title(title, fontsize=14.0, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=26, ha='right', rotation_mode='anchor', fontsize=14.8)
        ax.tick_params(axis='x', pad=0.3)
        ax.tick_params(axis='y', labelsize=9.5)
        ax.grid(axis='y', linestyle='--', alpha=0.35)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, val + 0.020, f'{val:.3f}', ha='center', va='bottom', fontsize=9.2)
    top_axes[0].text(-0.16, 1.06, '(a)', transform=top_axes[0].transAxes, fontsize=15.0, fontweight='bold')

    # (f) radar
    ax_r = fig.add_subplot(gs[0, 3], projection='polar')
    radar_norm = make_radar_norm()
    angles = np.linspace(0, 2 * np.pi, len(RADAR_AXES), endpoint=False)
    closed_angles = np.concatenate([angles, [angles[0]]])

    for i, m in enumerate(RADAR_METHODS):
        vals = np.concatenate([radar_norm[i], [radar_norm[i, 0]]])
        ax_r.plot(closed_angles, vals, color=RADAR_COLORS[m], linewidth=1.85, label=m)
        ax_r.fill(closed_angles, vals, color=RADAR_COLORS[m], alpha=0.055)

    ax_r.set_theta_offset(np.pi / 2)
    ax_r.set_theta_direction(-1)

    # Axis labels are pushed outward so that they do not touch the outer ring.
    ax_r.set_xticks(angles)
    ax_r.set_xticklabels(RADAR_AXES, fontsize=17.5)
    ax_r.tick_params(axis='x', pad=16)

    # Keep multiple rings, but hide radial numeric tick labels.
    # You can add the numbers manually later in PPT/PS if needed.
    ax_r.set_ylim(0.0, 1.05)
    ring_ticks = [0.2, 0.4, 0.6, 0.8, 1.0]
    ax_r.set_yticks(ring_ticks)
    ax_r.set_yticklabels([])

    ax_r.yaxis.grid(True, alpha=0.50, linestyle='--', linewidth=0.75)
    ax_r.xaxis.grid(True, alpha=0.35, linestyle='--', linewidth=0.65)
    ax_r.spines['polar'].set_alpha(0.35)

    ax_r.set_title('Baseline overview (raw values)', fontsize=12.0, fontweight='bold', pad=18)

    # Move legend to the upper-right outside the radar polygon.
    ax_r.legend(
        loc='upper right',
        bbox_to_anchor=(1.36, 1.18),
        fontsize=12.5,
        frameon=True,
        ncol=1,
        borderpad=0.38,
        handlelength=1.5
    )

    ax_r.text(-0.14, 1.08, '(b)', transform=ax_r.transAxes, fontsize=15.0, fontweight='bold')

    # (b)
    ax_b = fig.add_subplot(gs[1, 0:2])
    sns.heatmap(
        heat_df,
        ax=ax_b,
        cmap='RdBu_r',
        annot=True,
        fmt='.3f',
        linewidths=0.5,
        cbar_kws={'label': 'Average Precision'},
        annot_kws={'fontsize': 17.1},
    )
    ax_b.set_title('Top-12 models across metrics (Average Precision)', fontsize=14.0, fontweight='bold')
    ax_b.set_xlabel('')
    ax_b.set_ylabel('')
    style_heatmap_xticklabels(ax_b, fontsize=18.2)
    ax_b.tick_params(axis='y', labelsize=9.3)
    ax_b.text(-0.11, 1.06, '(c)', transform=ax_b.transAxes, fontsize=15.0, fontweight='bold')

    # (c)
    ax_c = fig.add_subplot(gs[1, 2:4])
    sns.violinplot(
        data=violin_df,
        x='MetricShow', y='AUROC', order=METRIC_SHOW,
        cut=0, inner='box', linewidth=0.8, ax=ax_c, color='#d9e2f3'
    )
    means = violin_df.groupby('MetricShow')['AUROC'].mean().reindex(METRIC_SHOW)
    ax_c.scatter(range(len(METRIC_SHOW)), means.values, marker='D', s=38, color='black', label='Mean', zorder=3)
    ax_c.set_title('Across-model AUROC distribution by metric', fontsize=14.0, fontweight='bold')
    ax_c.set_xlabel('')
    ax_c.set_ylabel('AUROC', fontsize=22.8)
    ax_c.tick_params(axis='x', rotation=20, labelsize=10.0)
    ax_c.legend(frameon=False, loc='upper left', fontsize=9.5)
    ax_c.text(-0.10, 1.06, '(d)', transform=ax_c.transAxes, fontsize=15.0, fontweight='bold')

    # (d)
    ax_d = fig.add_subplot(gs[2, 0:2])
    y = np.arange(len(ap_wide))
    point_colors = ['#cf4e4e' if d > 0 else '#7f8fa6' for d in ap_wide['Delta']]
    ax_d.hlines(y, 0, ap_wide['Delta'], color=point_colors, linewidth=2)
    ax_d.scatter(ap_wide['Delta'], y, color=point_colors, s=34, zorder=3)
    ax_d.axvline(0, color='#666666', linewidth=0.9)
    x_min = float(ap_wide['Delta'].min()) - 0.014
    x_max = float(ap_wide['Delta'].max()) + 0.014
    ax_d.set_xlim(x_min, x_max)
    for yi, dv in zip(y, ap_wide['Delta']):
        xt = dv + 0.0022 if dv >= 0 else dv - 0.0022
        ha = 'left' if dv >= 0 else 'right'
        ax_d.text(
            xt, yi, f'{dv:+.3f}', va='center', ha=ha, fontsize=8.2, color='#444444',
            bbox=dict(facecolor='white', edgecolor='none', pad=0.18, alpha=0.90), clip_on=True
        )
    ax_d.set_yticks(y)
    ax_d.set_yticklabels(ap_wide['ModelShort'], fontsize=12.2)
    ax_d.tick_params(axis='y', pad=6)
    ax_d.set_xlabel('AP gain of Ours-TCUS over the best non-TCUS metric', fontsize=20.9)
    ax_d.set_title('Per-model advantage of Ours-TCUS', fontsize=14.0, fontweight='bold')
    ax_d.grid(axis='x', linestyle='--', alpha=0.35)
    ax_d.text(-0.10, 1.06, '(e)', transform=ax_d.transAxes, fontsize=15.0, fontweight='bold')

    # (e)
    ax_e = fig.add_subplot(gs[2, 2:4])
    texts = []
    for provider, sub_df in scatter_df.groupby('Provider'):
        ax_e.scatter(
            sub_df['AUROC'],
            sub_df['Average Precision'],
            s=85 + 170 * sub_df['Task success ratio'],
            alpha=0.78,
            label=provider,
            color=PROVIDER_COLORS.get(provider, 'gray'),
            edgecolor='white',
            linewidth=0.7,
            zorder=3,
        )
    for _, row in scatter_df.iterrows():
        texts.append(ax_e.text(row['AUROC'] + 0.001, row['Average Precision'] + 0.001, row['ModelShort'], fontsize=8.2, color='#333333', zorder=4))
    ax_e.set_xlim(scatter_df['AUROC'].min() - 0.008, scatter_df['AUROC'].max() + 0.014)
    ax_e.set_ylim(scatter_df['Average Precision'].min() - 0.045, scatter_df['Average Precision'].max() + 0.045)
    if adjust_text is not None:
        adjust_text(
            texts,
            ax=ax_e,
            expand_text=(1.12, 1.25),
            expand_points=(1.10, 1.20),
            force_text=(0.35, 0.55),
            force_points=(0.18, 0.28),
            arrowprops=dict(arrowstyle='-', color='#777777', lw=0.35, alpha=0.60),
        )
    ax_e.set_xlabel('AUROC of Ours-TCUS', fontsize=20.9)
    ax_e.set_ylabel('Average Precision of Ours-TCUS', fontsize=20.9)
    ax_e.set_title('Trade-off under Ours-TCUS', fontsize=14.0, fontweight='bold')
    ax_e.grid(True, linestyle='--', alpha=0.35)
    from matplotlib.lines import Line2D

    provider_order = ['claude', 'gemini', 'gpt', 'grok', 'qwen']
    provider_labels = {
        'claude': 'Claude',
        'gemini': 'Gemini',
        'gpt': 'GPT',
        'grok': 'Grok',
        'qwen': 'Qwen',
    }
    legend_handles = [
        Line2D(
            [0], [0],
            marker='o',
            linestyle='None',
            label=provider_labels.get(k, k),
            markerfacecolor=PROVIDER_COLORS.get(k, 'gray'),
            markeredgecolor='white',
            markeredgewidth=0.8,
            markersize=21.8,
            alpha=0.90,
        )
        for k in provider_order
        if k in set(scatter_df['Provider'])
    ]

    ax_e.legend(
        handles=legend_handles,
        title='Provider',
        loc='upper left',
        bbox_to_anchor=(0.012, 0.988),
        ncol=3,
        frameon=True,
        fontsize=13.7,
        title_fontsize=28.1,
        borderpad=0.38,
        labelspacing=0.45,
        handletextpad=0.50,
        columnspacing=1.10,
        borderaxespad=0.32,
    )
    ax_e.text(-0.10, 1.06, '(f)', transform=ax_e.transAxes, fontsize=15.0, fontweight='bold')

    fig.suptitle('Task-Utility IQA: Complementary Views', fontsize=20.0, fontweight='bold', y=0.972)
    fig.subplots_adjust(top=0.92, left=0.08, right=0.965, bottom=0.06)
    # DIRECT_LAYOUT_FIX_FOR_FONT190
    fig.subplots_adjust(left=0.070, right=0.988, top=0.900, bottom=0.075, wspace=0.30, hspace=0.68)
    fig.savefig(os.path.join(outdir, 'task_utility_composite_main.png'), bbox_inches='tight')
    fig.savefig(os.path.join(outdir, 'task_utility_composite_main.pdf'), bbox_inches='tight')
    plt.close(fig)


def main():
    style_setup()
    if not os.path.exists(INPUT_XLSX):
        raise FileNotFoundError(f'Cannot find {INPUT_XLSX} in the current directory.')
    os.makedirs(OUTDIR, exist_ok=True)
    long_df = load_long_summary(INPUT_XLSX)
    make_baseline_heatmap(OUTDIR)
    make_composite(long_df, OUTDIR)
    print('Done.')
    print(f'Output directory: {OUTDIR}')
    for name in [
        'baseline_metric_matrix_no_valid_ratio.png',
        'baseline_metric_matrix_no_valid_ratio.pdf',
        'task_utility_composite_main.png',
        'task_utility_composite_main.pdf',
        'per_model_ap_gain.csv',
    ]:
        print(' -', os.path.join(OUTDIR, name))


if __name__ == '__main__':
    main()
