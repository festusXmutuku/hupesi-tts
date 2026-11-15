import type { VoiceOption } from "./types";

// Descriptions from /voices have the form "Nam · Bac · Phong cach tin tuc · 48 kHz".
// Reading that as "Male · North" (gender · region) is confusing, so we reformat it as
// "Male, Northern · News" for plain-text selects.

// The third part is the speaking style, recorded in two forms:
// "Phong c\u00e1ch tin t\u1ee9c" or "Voice t\u1ef1 nhi\u00ean" (Ngoc Huyen, Adam).
const STYLE_PREFIX = /^(phong c\u00e1ch|voice)\s+/i;
const KHZ = /\bkHz\b/i;

export function voiceStyle(v: VoiceOption): string {
  const parts = (v.description ?? "").split("·").map((p) => p.trim());
  const part = parts.length >= 3 ? parts[2] : "";
  if (!part || KHZ.test(part)) {
    return "";
  }
  const style = part.replace(STYLE_PREFIX, "");
  return style.charAt(0).toUpperCase() + style.slice(1);
}

/** "Male, Northern · News" - empty when there is no information. */
export function voiceDetail(v: VoiceOption): string {
  if (v.isCloned) {
    return "Cloned voice";
  }
  const who: string[] = [];
  if (v.gender === "male") {
    who.push("Male");
  } else if (v.gender === "female") {
    who.push("Female");
  }
  if (v.region === "North") {
    who.push("Northern");
  } else if (v.region === "South") {
    who.push("Southern");
  } else if (v.region === "Central") {
    who.push("Central");
  }
  const style = voiceStyle(v);
  return [who.join(", "), style].filter(Boolean).join(" · ");
}

export function voiceOptionLabel(v: VoiceOption): string {
  const detail = voiceDetail(v);
  return detail ? `${v.name} - ${detail}` : v.name;
}

export interface VoiceGroup {
  label: string;
  voices: VoiceOption[];
}

/** Groups for <optgroup>: Female, Male, Cloned, Other - omits empty groups and preserves original order within each group. */
export function groupVoices(voices: VoiceOption[]): VoiceGroup[] {
  const groups: VoiceGroup[] = [
    { label: "Female voices", voices: [] },
    { label: "Male voices", voices: [] },
    { label: "Cloned voices", voices: [] },
    { label: "Other", voices: [] },
  ];
  for (const v of voices) {
    if (v.isCloned) {
      groups[2].voices.push(v);
    } else if (v.gender === "female") {
      groups[0].voices.push(v);
    } else if (v.gender === "male") {
      groups[1].voices.push(v);
    } else {
      groups[3].voices.push(v);
    }
  }
  return groups.filter((g) => g.voices.length > 0);
}
