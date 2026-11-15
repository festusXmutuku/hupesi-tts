import { Info } from "lucide-react";
import type React from "react";
import { useEffect, useId, useLayoutEffect, useRef, useState } from "react";

interface InfoTipProps {
  children: React.ReactNode;
  /** Option name, used for the screen reader label: "Explain <label>". */
  label: string;
}

/**
 * An ⓘ button that opens an explanation panel just below its row (the row needs `position: relative`).
 * Uses click instead of hover because `title` tooltips appear slowly, can be clipped, and don't work on touch screens.
 */
export function InfoTip({ label, children }: InfoTipProps) {
  const [open, setOpen] = useState(false);
  const panelId = useId();
  const rootRef = useRef<HTMLSpanElement>(null);
  const panelRef = useRef<HTMLSpanElement>(null);
  const [above, setAbove] = useState(false);

  // If the row is near the bottom of the window (e.g. "Stable"), open upward when there isn't enough space below.
  useLayoutEffect(() => {
    if (!(open && panelRef.current)) {
      setAbove(false);
      return;
    }
    const rect = panelRef.current.getBoundingClientRect();
    setAbove(rect.bottom > window.innerHeight - 8);
  }, [open]);

  useEffect(() => {
    if (!open) {
      return;
    }
    const onPointer = (e: PointerEvent) => {
      if (!rootRef.current?.contains(e.target as Node)) {
        setOpen(false);
      }
    };
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") {
        e.stopPropagation();
        setOpen(false);
      }
    };
    document.addEventListener("pointerdown", onPointer);
    document.addEventListener("keydown", onKey, true);
    return () => {
      document.removeEventListener("pointerdown", onPointer);
      document.removeEventListener("keydown", onKey, true);
    };
  }, [open]);

  return (
    <span className="info-tip" ref={rootRef}>
      <button
        aria-controls={panelId}
        aria-expanded={open}
        aria-label={`Explain ${label}`}
        className="info-tip-button"
        onClick={(e) => {
          // The button is inside a switch's <label>: clicking it must not toggle the switch.
          e.preventDefault();
          e.stopPropagation();
          setOpen((v) => !v);
        }}
        type="button"
      >
        <Info size={15} />
      </button>
      {open && (
        <span
          className="info-tip-panel"
          data-above={above || undefined}
          id={panelId}
          ref={panelRef}
          role="note"
        >
          {children}
        </span>
      )}
    </span>
  );
}
