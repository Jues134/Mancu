/** 用户说了时长就听他的，没说他再按字数估。
 *
 * 抽成模块是为了能单独量：这段逻辑错了不会报错，只会让整部片的目标时长
 * 差一截——「1 分钟左右」被估成 30 秒，界面还理直气壮显示「目标 30 秒」。
 */

/** 中文字数写法，只覆盖时长里真会出现的那些。 */
const CN_NUMBER: Record<string, number> = {
  一: 1, 二: 2, 两: 2, 三: 3, 四: 4, 五: 5,
  六: 6, 七: 7, 八: 8, 九: 9, 十: 10, 半: 0.5,
};

/** 「1 分钟左右」「两分半」「90 秒」→ 秒数。没提到时长返回 null。 */
export function statedSeconds(text: string): number | null {
  const m = text.match(/(\d+(?:\.\d+)?|[一二三四五六七八九十两半]{1,3})\s*(秒钟|秒|分钟|分)/);
  if (!m) return null;
  const raw = m[1];
  let value = Number(raw);
  if (Number.isNaN(value)) {
    if (raw.length === 1) {
      value = CN_NUMBER[raw] ?? NaN;
    } else if (raw.length === 2 && raw[0] === "十") {
      value = 10 + (CN_NUMBER[raw[1]] ?? 0);
    } else if (raw.length === 2 && raw[1] === "十") {
      value = (CN_NUMBER[raw[0]] ?? 0) * 10;
    } else if (raw.length === 3 && raw[1] === "十") {
      value = (CN_NUMBER[raw[0]] ?? 0) * 10 + (CN_NUMBER[raw[2]] ?? 0);
    } else {
      // 「两分半」这种后面还挂着「半」的，取前半部分就够了
      value = CN_NUMBER[raw[0]] ?? NaN;
    }
  }
  if (!Number.isFinite(value) || value <= 0) return null;
  const seconds = m[2].startsWith("分") ? value * 60 : value;
  // 3 秒到 5 分钟。上限是产品口径，不是技术限制：再长就该分段，而不是一镜到底。
  return Math.round(Math.min(300, Math.max(3, seconds)));
}

/** How long this piece of text should run.
 *
 * 只有**短请求**里说的时长才当真。长正文里的「得 5 分」「每 3 分钟一班」是正文
 * 内容，不是对片长的要求；照它算会把一部五分钟的片子压成 5 秒。判定线跟后端
 * 区分「主题式请求」和「一段材料」用的是同一个量级（160 字）。
 *
 * 估的是「旁白念一遍要多久」，动作与转场会再往上加，所以这是下限而不是承诺。
 */
export function targetSecondsFor(text: string): number {
  const trimmed = text.trim();
  if (trimmed.length <= 160) {
    const stated = statedSeconds(trimmed);
    if (stated !== null) return stated;
  }
  return Math.max(30, Math.min(300, Math.round(trimmed.length / 4.5)));
}
