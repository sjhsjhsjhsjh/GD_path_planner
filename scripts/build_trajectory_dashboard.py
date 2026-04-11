import argparse
import csv
import json
import os

import numpy as np


def _safe_float(row, key, default=0.0):
    try:
        return float(row.get(key, default) or default)
    except Exception:
        return default


def _safe_int(row, key, default=0):
    try:
        return int(float(row.get(key, default) or default))
    except Exception:
        return default


def _load_rows(csv_path):
    with open(csv_path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _load_train_rewards(run_dir, step_csv_path=None):
    rewards = {}
    candidate_paths = []

    if step_csv_path:
        step_name = os.path.basename(step_csv_path)
        if step_name.endswith("_step_rewards.csv"):
            prefix = step_name[: -len("_step_rewards.csv")]
            candidate_paths.append(os.path.join(run_dir, f"{prefix}_train_log.csv"))

    candidate_paths.append(os.path.join(run_dir, "train_log.csv"))

    train_log_path = None
    for path in candidate_paths:
        if os.path.exists(path):
            train_log_path = path
            break
    if train_log_path is None:
        return rewards

    with open(train_log_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rewards[_safe_int(row, "episode")] = _safe_float(row, "total_reward")
    return rewards


def _prepare_episode_data(rows, train_rewards):
    episodes = {}
    for row in rows:
        episode = _safe_int(row, "episode")
        item = {
            "step": _safe_int(row, "step"),
            "action": _safe_int(row, "action"),
            "last_x": _safe_int(row, "last_pos_x"),
            "last_y": _safe_int(row, "last_pos_y"),
            "x": _safe_int(row, "pos_x"),
            "y": _safe_int(row, "pos_y"),
            "goal_x": _safe_int(row, "goal_x"),
            "goal_y": _safe_int(row, "goal_y"),
            "ins_error": _safe_int(row, "ins_error"),
            "step_reward": _safe_float(row, "step_reward"),
            "terminated": _safe_int(row, "terminated"),
            "termination_reason": row.get("termination_reason", "unknown"),
        }
        episodes.setdefault(episode, []).append(item)

    result = {}
    for episode, items in sorted(episodes.items()):
        items.sort(key=lambda item: item["step"])
        start = {"x": items[0]["last_x"], "y": items[0]["last_y"]}
        goal = {"x": items[0]["goal_x"], "y": items[0]["goal_y"]}
        points = [{"x": start["x"], "y": start["y"], "ins_error": 0, "step": 0}]
        for item in items:
            points.append(
                {
                    "x": item["x"],
                    "y": item["y"],
                    "ins_error": item["ins_error"],
                    "step": item["step"],
                }
            )

        result[episode] = {
            "episode": episode,
            "start": start,
            "goal": goal,
            "points": points,
            "steps": len(items),
            "final_ins_error": items[-1]["ins_error"],
            "max_ins_error": max(point["ins_error"] for point in points),
            "termination_reason": items[-1]["termination_reason"],
            "terminated": items[-1]["terminated"],
            "total_reward": train_rewards.get(
                episode - 1, sum(item["step_reward"] for item in items)
            ),
        }
    return result


def build_dashboard(csv_path, output_path=None):
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV 文件不存在: {csv_path}")

    run_dir = os.path.dirname(csv_path)
    rows = _load_rows(csv_path)
    if not rows:
        raise ValueError("step_rewards.csv 中没有数据")

    terrain_path = os.path.join(run_dir, "seafloor.npy")
    beacon_path = os.path.join(run_dir, "beacon_map.npy")
    terrain = np.load(terrain_path).tolist() if os.path.exists(terrain_path) else None
    beacon_map = np.load(beacon_path).tolist() if os.path.exists(beacon_path) else None
    train_rewards = _load_train_rewards(run_dir, step_csv_path=csv_path)
    episodes = _prepare_episode_data(rows, train_rewards)

    max_ins_error = 0
    for episode in episodes.values():
        max_ins_error = max(max_ins_error, episode["max_ins_error"])

    payload = {
        "terrain": terrain,
        "beaconMap": beacon_map,
        "episodes": episodes,
        "maxInsError": max_ins_error,
    }

    if output_path is None:
        output_path = os.path.join(run_dir, "reward_plots", "trajectory_dashboard.html")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    html = f"""<!DOCTYPE html>
<html lang=\"zh-CN\">
<head>
  <meta charset=\"UTF-8\" />
  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1.0\" />
  <title>Trajectory Dashboard</title>
  <style>
    :root {{
      --bg: #f4efe6;
      --panel: rgba(255,255,255,0.78);
      --ink: #1d1b1a;
      --muted: #6f6a64;
      --accent: #6a00ff;
      --accent2: #ff3b3b;
    }}
    body {{
      margin: 0;
      font-family: "Segoe UI", "Microsoft YaHei", sans-serif;
      background:
        radial-gradient(circle at top left, rgba(255, 192, 203, 0.28), transparent 32%),
        radial-gradient(circle at bottom right, rgba(106, 0, 255, 0.18), transparent 28%),
        linear-gradient(135deg, #f8f2ea, #efe6db 55%, #e4dbd0);
      color: var(--ink);
      min-height: 100vh;
    }}
    .layout {{
      display: grid;
      grid-template-columns: 360px 1fr;
      gap: 24px;
      padding: 24px;
    }}
    .panel {{
      background: var(--panel);
      backdrop-filter: blur(10px);
      border: 1px solid rgba(255,255,255,0.7);
      border-radius: 20px;
      box-shadow: 0 16px 40px rgba(0,0,0,0.08);
    }}
    .sidebar {{
      padding: 22px;
      align-self: start;
      position: sticky;
      top: 20px;
    }}
    h1 {{
      font-size: 28px;
      margin: 0 0 8px;
    }}
    .sub {{
      color: var(--muted);
      line-height: 1.55;
      margin-bottom: 22px;
    }}
    label {{
      display: block;
      margin-bottom: 8px;
      font-weight: 600;
    }}
    select, button {{
      width: 100%;
      border: 0;
      border-radius: 14px;
      padding: 12px 14px;
      font-size: 15px;
      background: white;
      box-shadow: inset 0 0 0 1px rgba(0,0,0,0.08);
    }}
    .row {{
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 10px;
      margin-top: 10px;
    }}
    .summary {{
      margin-top: 18px;
      display: grid;
      gap: 10px;
    }}
    .metric {{
      padding: 12px 14px;
      border-radius: 14px;
      background: rgba(255,255,255,0.72);
    }}
    .metric .k {{
      color: var(--muted);
      font-size: 12px;
      margin-bottom: 4px;
      text-transform: uppercase;
      letter-spacing: 0.08em;
    }}
    .metric .v {{
      font-size: 20px;
      font-weight: 700;
    }}
    .canvas-wrap {{
      padding: 18px;
      display: flex;
      flex-direction: column;
      gap: 12px;
    }}
    canvas {{
      width: 100%;
      aspect-ratio: 1 / 1;
      border-radius: 18px;
      background: rgba(255,255,255,0.85);
      box-shadow: inset 0 0 0 1px rgba(0,0,0,0.08);
    }}
    .legend {{
      display: flex;
      align-items: center;
      gap: 12px;
      color: var(--muted);
      font-size: 13px;
    }}
    .legend-bar {{
      height: 14px;
      flex: 1;
      border-radius: 999px;
      background: linear-gradient(90deg, #ff3b3b, #d81b60, #9c27b0, #6a00ff);
    }}
    .notes {{
      color: var(--muted);
      font-size: 13px;
      line-height: 1.5;
    }}
    @media (max-width: 980px) {{
      .layout {{ grid-template-columns: 1fr; }}
      .sidebar {{ position: static; }}
    }}
  </style>
</head>
<body>
  <div class=\"layout\">
    <aside class=\"panel sidebar\">
      <h1>Trajectory Inspector</h1>
      <div class=\"sub\">按 episode 查看轨迹。线段颜色直接映射 INS_error，颜色越偏紫，说明惯导误差越高。</div>
      <label for=\"episodeSelect\">选择 Episode</label>
      <select id=\"episodeSelect\"></select>
      <div class=\"row\">
        <button id=\"prevBtn\">上一个</button>
        <button id=\"nextBtn\">下一个</button>
      </div>
      <div class=\"summary\" id=\"summary\"></div>
      <div class=\"notes\">起点为绿色圆点，终点为金色星形，青色半透明网格表示信标覆盖区域。</div>
    </aside>
    <main class=\"panel canvas-wrap\">
      <canvas id=\"canvas\" width=\"900\" height=\"900\"></canvas>
      <div class=\"legend\">
        <span>INS 低</span>
        <div class=\"legend-bar\"></div>
        <span>INS 高</span>
      </div>
    </main>
  </div>
  <script>
    const payload = {json.dumps(payload, ensure_ascii=False)};
    const canvas = document.getElementById('canvas');
    const ctx = canvas.getContext('2d');
    const select = document.getElementById('episodeSelect');
    const summary = document.getElementById('summary');
    const episodeKeys = Object.keys(payload.episodes).map(Number).sort((a, b) => a - b);
    let currentEpisode = episodeKeys[0];

    function lerp(a, b, t) {{
      return a + (b - a) * t;
    }}

    function colorForIns(insError) {{
      const maxValue = Math.max(payload.maxInsError, 1);
      const t = Math.max(0, Math.min(1, insError / maxValue));
      const stops = [
        [255, 59, 59],
        [216, 27, 96],
        [156, 39, 176],
        [106, 0, 255],
      ];
      const scaled = t * (stops.length - 1);
      const idx = Math.min(stops.length - 2, Math.floor(scaled));
      const localT = scaled - idx;
      const c1 = stops[idx];
      const c2 = stops[idx + 1];
      const r = Math.round(lerp(c1[0], c2[0], localT));
      const g = Math.round(lerp(c1[1], c2[1], localT));
      const b = Math.round(lerp(c1[2], c2[2], localT));
      return `rgb(${{r}}, ${{g}}, ${{b}})`;
    }}

    function terrainColor(v) {{
      const value = Math.max(0, Math.min(1, v));
      const r = Math.round(lerp(238, 49, value));
      const g = Math.round(lerp(232, 118, value));
      const b = Math.round(lerp(223, 141, value));
      return `rgb(${{r}}, ${{g}}, ${{b}})`;
    }}

    function toCanvasPoint(x, y, cols, rows, margin, size) {{
      const cell = size / Math.max(cols, rows);
      const drawWidth = cols * cell;
      const drawHeight = rows * cell;
      const offsetX = margin + (size - drawWidth) / 2;
      const offsetY = margin + (size - drawHeight) / 2;
      return {{
        x: offsetX + (x + 0.5) * cell,
        y: offsetY + drawHeight - (y + 0.5) * cell,
        cell,
        offsetX,
        offsetY,
        drawWidth,
        drawHeight,
      }};
    }}

    function drawBackground(cols, rows, margin, size) {{
      ctx.fillStyle = 'rgba(255,255,255,0.9)';
      ctx.fillRect(0, 0, canvas.width, canvas.height);

      if (!payload.terrain) return;

      const cell = size / Math.max(cols, rows);
      const drawWidth = cols * cell;
      const drawHeight = rows * cell;
      const offsetX = margin + (size - drawWidth) / 2;
      const offsetY = margin + (size - drawHeight) / 2;

      for (let x = 0; x < cols; x++) {{
        for (let y = 0; y < rows; y++) {{
          const v = payload.terrain[x][y];
          ctx.fillStyle = terrainColor(v);
          ctx.fillRect(offsetX + x * cell, offsetY + (rows - y - 1) * cell, cell + 0.5, cell + 0.5);
        }}
      }}

      if (payload.beaconMap) {{
        for (let x = 0; x < cols; x++) {{
          for (let y = 0; y < rows; y++) {{
            if (payload.beaconMap[x][y] > 0) {{
              ctx.fillStyle = 'rgba(0, 204, 255, 0.18)';
              ctx.fillRect(offsetX + x * cell, offsetY + (rows - y - 1) * cell, cell + 0.5, cell + 0.5);
            }}
          }}
        }}
      }}

      ctx.strokeStyle = 'rgba(0,0,0,0.08)';
      ctx.lineWidth = 1;
      ctx.strokeRect(offsetX, offsetY, drawWidth, drawHeight);
    }}

    function drawEpisode(episodeKey) {{
      const episode = payload.episodes[episodeKey];
      if (!episode) return;
      const cols = payload.terrain ? payload.terrain.length : 50;
      const rows = payload.terrain ? payload.terrain[0].length : 50;
      const margin = 28;
      const size = canvas.width - margin * 2;

      drawBackground(cols, rows, margin, size);

      const pts = episode.points;
      for (let i = 1; i < pts.length; i++) {{
        const p0 = toCanvasPoint(pts[i - 1].x, pts[i - 1].y, cols, rows, margin, size);
        const p1 = toCanvasPoint(pts[i].x, pts[i].y, cols, rows, margin, size);
        ctx.beginPath();
        ctx.moveTo(p0.x, p0.y);
        ctx.lineTo(p1.x, p1.y);
        ctx.strokeStyle = colorForIns(pts[i].ins_error);
        ctx.lineWidth = 4;
        ctx.lineCap = 'round';
        ctx.stroke();
      }}

      for (const point of pts) {{
        const p = toCanvasPoint(point.x, point.y, cols, rows, margin, size);
        ctx.beginPath();
        ctx.fillStyle = colorForIns(point.ins_error);
        ctx.arc(p.x, p.y, 4.2, 0, Math.PI * 2);
        ctx.fill();
      }}

      const start = toCanvasPoint(episode.start.x, episode.start.y, cols, rows, margin, size);
      ctx.beginPath();
      ctx.fillStyle = '#22c55e';
      ctx.arc(start.x, start.y, 8, 0, Math.PI * 2);
      ctx.fill();
      ctx.strokeStyle = '#111';
      ctx.lineWidth = 1.5;
      ctx.stroke();

      const goal = toCanvasPoint(episode.goal.x, episode.goal.y, cols, rows, margin, size);
      const spikes = 5;
      const outerRadius = 11;
      const innerRadius = 5;
      let rot = Math.PI / 2 * 3;
      ctx.beginPath();
      ctx.moveTo(goal.x, goal.y - outerRadius);
      for (let i = 0; i < spikes; i++) {{
        ctx.lineTo(goal.x + Math.cos(rot) * outerRadius, goal.y + Math.sin(rot) * outerRadius);
        rot += Math.PI / spikes;
        ctx.lineTo(goal.x + Math.cos(rot) * innerRadius, goal.y + Math.sin(rot) * innerRadius);
        rot += Math.PI / spikes;
      }}
      ctx.closePath();
      ctx.fillStyle = '#ffd166';
      ctx.fill();
      ctx.strokeStyle = '#111';
      ctx.stroke();

      summary.innerHTML = `
        <div class=\"metric\"><div class=\"k\">Episode</div><div class=\"v\">${{episode.episode}}</div></div>
        <div class=\"metric\"><div class=\"k\">Total Reward</div><div class=\"v\">${{episode.total_reward.toFixed(3)}}</div></div>
        <div class=\"metric\"><div class=\"k\">Steps</div><div class=\"v\">${{episode.steps}}</div></div>
        <div class=\"metric\"><div class=\"k\">Termination</div><div class=\"v\">${{episode.termination_reason}}</div></div>
        <div class=\"metric\"><div class=\"k\">Final INS</div><div class=\"v\">${{episode.final_ins_error}}</div></div>
        <div class=\"metric\"><div class=\"k\">Max INS</div><div class=\"v\">${{episode.max_ins_error}}</div></div>
      `;
    }}

    function updateSelection(value) {{
      currentEpisode = Number(value);
      select.value = String(currentEpisode);
      drawEpisode(currentEpisode);
    }}

    for (const ep of episodeKeys) {{
      const option = document.createElement('option');
      option.value = String(ep);
      option.textContent = `Episode ${{ep}}`;
      select.appendChild(option);
    }}

    select.addEventListener('change', (event) => updateSelection(event.target.value));
    document.getElementById('prevBtn').addEventListener('click', () => {{
      const idx = Math.max(0, episodeKeys.indexOf(currentEpisode) - 1);
      updateSelection(episodeKeys[idx]);
    }});
    document.getElementById('nextBtn').addEventListener('click', () => {{
      const idx = Math.min(episodeKeys.length - 1, episodeKeys.indexOf(currentEpisode) + 1);
      updateSelection(episodeKeys[idx]);
    }});

    updateSelection(currentEpisode);
  </script>
</body>
</html>
"""

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)

    return output_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True, help="step_rewards.csv 的路径")
    parser.add_argument("--output", help="输出 HTML 路径")
    args = parser.parse_args()
    out_path = build_dashboard(args.csv, output_path=args.output)
    print(f"trajectory dashboard 已输出到: {out_path}")
