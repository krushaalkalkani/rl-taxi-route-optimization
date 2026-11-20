import "server-only";
import { readFileSync, existsSync } from "node:fs";
import path from "node:path";

const DIR = path.join(process.cwd(), "public", "data");

export function readJson<T>(name: string): T | null {
  const p = path.join(DIR, name);
  if (!existsSync(p)) return null;
  return JSON.parse(readFileSync(p, "utf8")) as T;
}
