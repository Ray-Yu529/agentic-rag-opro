// best-objective-so-far 曲線: 比較三策略「用多少評估次數爬到多高」
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer,
} from "recharts";

const COLORS = { random: "#888888", opro: "#d1495b", hybrid: "#2e7d32" };

export default function ObjectiveChart({ strategies }) {
  const names = Object.keys(strategies || {});
  if (names.length === 0) return null;

  const maxLen = Math.max(...names.map((n) => strategies[n].curve.length));
  const data = Array.from({ length: maxLen }, (_, i) => {
    const row = { x: i + 1 };
    names.forEach((n) => {
      const c = strategies[n].curve;
      row[n] = i < c.length ? Number(c[i].toFixed(3)) : null;
    });
    return row;
  });

  return (
    <div className="card">
      <h2>最佳化效率（越快越高越好）</h2>
      <ResponsiveContainer width="100%" height={280}>
        <LineChart data={data} margin={{ top: 10, right: 20, bottom: 10, left: -10 }}>
          <CartesianGrid strokeDasharray="3 3" opacity={0.3} />
          <XAxis dataKey="x" label={{ value: "評估次數", position: "insideBottom", offset: -5 }} />
          <YAxis domain={["auto", "auto"]} />
          <Tooltip />
          <Legend />
          {names.map((n) => (
            <Line key={n} type="stepAfter" dataKey={n} stroke={COLORS[n] || "#3366cc"}
                  strokeWidth={2} dot={{ r: 3 }} connectNulls />
          ))}
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}
