// best-objective-so-far 曲線: 比較三策略「用多少評估次數爬到多高」
// 系列色經 CVD/對比驗證: opro=blue, hybrid=aqua, random=中性灰 (baseline)
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer,
} from "recharts";

const COLORS = { random: "#898781", opro: "#2a78d6", hybrid: "#1baf7a" };
const AXIS = { fontSize: 12, fill: "#898781" };

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
      <h2>最佳化效率</h2>
      <p className="chart-note">best objective so far，越快爬越高越好</p>
      <ResponsiveContainer width="100%" height={280}>
        <LineChart data={data} margin={{ top: 10, right: 20, bottom: 10, left: -10 }}>
          <CartesianGrid stroke="#e1e0d9" vertical={false} />
          <XAxis dataKey="x" tick={AXIS} tickLine={false}
                 axisLine={{ stroke: "#c3c2b7" }}
                 label={{ value: "評估次數", position: "insideBottom", offset: -5,
                          fontSize: 12, fill: "#898781" }} />
          <YAxis domain={["auto", "auto"]} tick={AXIS} tickLine={false}
                 axisLine={false} width={52} />
          <Tooltip cursor={{ stroke: "#c3c2b7", strokeDasharray: "4 4" }} />
          <Legend iconType="plainline" iconSize={14} />
          {names.map((n) => (
            /* 各策略可能用不同 λ 跑 (objective 定義不同)，標進圖例避免誤比 */
            <Line key={n} type="stepAfter" dataKey={n}
                  stroke={COLORS[n] || "#4a3aa7"} strokeWidth={2}
                  dot={{ r: 3, strokeWidth: 0, fill: COLORS[n] || "#4a3aa7" }}
                  activeDot={{ r: 5 }} connectNulls
                  name={strategies[n].lam != null ? `${n} (λ=${strategies[n].lam})` : n} />
          ))}
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}
