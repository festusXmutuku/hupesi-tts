import { useRef } from "react";
import { insertAtCursor } from "./text-tools";

/**
 * Inserts an emotion or pause tag at the current caret position in a text field
 * (replacing the selection, with the caret placed right after the tag). Previously,
 * tags were always appended to the end of the text (reported by a user, 2026-09).
 * Shared by TTS Studio and the Voice Cloning tab.
 */
export function useCaretInsert(
  text: string,
  setText: (value: string) => void,
  disabled = false
) {
  const textareaRef = useRef<HTMLTextAreaElement>(null);
  // If the text field has never been focused, the browser caret position is meaningless
  // (usually 0): in that case, append the tag to the end as before.
  const touchedRef = useRef(false);

  const onFocus = () => {
    touchedRef.current = true;
  };

  const insertTag = (tag: string) => {
    if (disabled) {
      return;
    }
    const el = textareaRef.current;
    const useCaret = el !== null && touchedRef.current;
    const start = useCaret ? el.selectionStart : text.length;
    const end = useCaret ? el.selectionEnd : text.length;
    const result = insertAtCursor(text, start, end, tag);
    setText(result.text);
    // Place the caret right after the tag so the user can insert another tag or keep typing in place.
    requestAnimationFrame(() => {
      const node = textareaRef.current;
      if (node) {
        node.focus();
        node.setSelectionRange(result.caret, result.caret);
        touchedRef.current = true;
      }
    });
  };

  return { textareaRef, onFocus, insertTag };
}
