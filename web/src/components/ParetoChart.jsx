// 正確率 vs 幻覺率 散點 + Pareto 前緣: 看多目標權衡
import {
  ScatterChart, Scatter, XAxis, YAxis, CartesianGrid, Tooltip, Legend,
  ResponsiveContainer, ZAxis,
} from "recharts";

export default function ParetoChart({ pareto }) {
  if (!pareto || !pareto.points || pareto.points.length === 0) return null;
  const toPct = (arr) => arr.map((p) => ({
    x: +(p.hallucination * 100).toFixed(1),
    y: +(p.correctness * 100).toFixed(1),
  }));
  const all = toPct(pareto.points);
  const front = toPct(pareto.front).sort((a, b) => a.x - b.x);

  return (
    <div className="card">
      <h2>多目標權衡：正確率 vs 幻覺率</h2>
      <ResponsiveContainer width="100%" height={280}>
        <ScatterChart margin={{ top: 10, right: 20, bottom: 15, left: -10 }}>
          <CartesianGrid strokeDasharray="3 3" opacity={0.3} />
          <XAxis type="number" dataKey="x" name="幻覺率"
                 label={{ value: "幻覺率 % (越低越好)", position: "insideBottom", offset: -8 }} />
          <YAxis type="number" dataKey="y" name="正確率"
                 label={{ value: "正確率 %", angle: -90, position: "insideLeft" }} />
          <ZAxis range={[60, 60]} />
          <Tooltip cursor={{ strokeDasharray: "3 3" }} />
          <Legend />
          <Scatter name="所有配置" data={all} fill="#bbbbbb" />
          <Scatter name="Pareto 前緣" data={front} fill="#d1495b" line shape="circle" />
        </ScatterChart>
      </ResponsiveContainer>
    </div>
  );
}
