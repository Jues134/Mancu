/** One-character avatar for an agent's name.
 *
 * `slice(0, 1)` is right for most of the roster (导演 → 导, 剧本分镜 → 剧) and
 * wrong for one: 一致性 starts with 一, which renders as a dash and reads as a
 * missing icon rather than an avatar. The fix is a lookup, not a redesign — the
 * name still shows next to it.
 */
const OVERRIDES: Record<string, string> = {
  // 一致性: 「一」looks like a placeholder. 「致」 is the word's own character.
  一致性: "致",
};

export function agentInitial(name: string): string {
  return OVERRIDES[name] || (name || "?").slice(0, 1);
}
