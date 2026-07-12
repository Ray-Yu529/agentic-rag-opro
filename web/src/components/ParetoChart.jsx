// 正確率 vs 幻覺率 散點 + Pareto 前緣: 三目標權衡 (第三軸 = 成本，點大小編碼)
// 所有配置 = 中性灰 (背景)，前緣 = 品牌藍 (焦點)
import {
  ScatterChart, Scatter, XAxis, YAxis, CartesianGrid, Tooltip, Legend,
  ResponsiveContainer, ZAxis,
} from "recharts";

const AXIS = { fontSize: 12, fill: "#898781" };

export default function ParetoChart({ pareto }) {
  if (!pareto || !pareto.points || pareto.points.length === 0) return null;
  const toPct = (arr) => arr.map((p) => ({
    x: +(p.hallucination * 100).toFixed(1),
    y: +(p.correctness * 100).toFixed(1),
    z: Math.round(p.tokens || 0),
  }));
  const all = toPct(pareto.points);
  const front = toPct(pareto.front).sort((a, b) => a.x - b.x);

  return (
    <div className="card">
      <h2>三目標權衡</h2>
      <p className="chart-note">正確率 vs 幻覺率，點大小 = 每題 token 成本；左上角的小點是理想區</p>
      <ResponsiveContainer width="100%" height={280}>
        <ScatterChart margin={{ top: 10, right: 20, bottom: 15, left: -10 }}>
          <CartesianGrid stroke="#e1e0d9" />
          <XAxis type="number" dataKey="x" name="幻覺率"
                 tick={AXIS} tickLine={false} axisLine={{ stroke: "#c3c2b7" }}
                 label={{ value: "幻覺率 % (越低越好)", position: "insideBottom",
                          offset: -8, fontSize: 12, fill: "#898781" }} />
          <YAxis type="number" dataKey="y" name="正確率"
                 tick={AXIS} tickLine={false} axisLine={false} width={52}
                 label={{ value: "正確率 %", angle: -90, position: "insideLeft",
                          fontSize: 12, fill: "#898781" }} />
          <ZAxis type="number" dataKey="z" name="每題 tokens" range={[45, 320]} />
          <Tooltip cursor={{ stroke: "#c3c2b7", strokeDasharray: "4 4" }} />
          <Legend iconSize={10} />
          <Scatter name="所有配置" data={all} fill="#c3c2b7" fillOpacity={0.7} />
          <Scatter name="Pareto 前緣" data={front} fill="#2a78d6"
                   line={{ stroke: "#2a78d6", strokeWidth: 2 }} shape="circle" />
        </ScatterChart>
      </ResponsiveContainer>
    </div>
  );
}
