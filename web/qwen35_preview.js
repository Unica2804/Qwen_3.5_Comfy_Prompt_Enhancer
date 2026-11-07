// Qwen3.5 Prompt Enhancer — streaming preview (4.x polish).
//
// The node pushes decoded partial text through PromptServer (`qwen35_preview`
// events) while it generates; this extension renders them into the node canvas
// overlay below it.  No socket ids are needed: send_sync without one broadcasts.
import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

const extension = {
    name: "qwen35.prompt-enhancer.stream-preview",

    setup() {
        api.addEventListener("qwen35_preview", (event) => {
            const detail = event.detail || {};
            if (!detail.text) return;
            // attach the partial text to the most recent enhancer node, if any
            const node = app.nodes?.find?.((n) => n.type === "Qwen35PromptEnhancer");
            if (!node) return;
            node._stream_preview = detail.text;
            if (typeof node.setDirtyCanvas === "function") node.setDirtyCanvas(true, true);
        });
    },

    nodeCreated(node) {
        if (node.type !== "Qwen35PromptEnhancer") return;
        const orig = node.onDrawForeground;
        node.onDrawForeground = function (ctx) {
            if (orig) orig.call(this, ctx);
            const text = this._stream_preview;
            if (!text) return;
            try {
                const size = node.size;
                const n = Math.min(4, String(text).split("\n").length);
                ctx.save();
                ctx.font = "11px monospace";
                ctx.fillStyle = "rgba(10,10,10,0.72)";
                ctx.fillRect(4, size[1] - 18 - n * 12, size[0] - 8, 14 + n * 12);
                ctx.fillStyle = "#cfe3ff";
                const lines = String(text).split("\n").slice(-n);
                let y = size[1] - 6 - (n - 1) * 12;
                for (const line of lines) {
                    ctx.fillText(line.slice(0, Math.floor((size[0] - 16) / 6.5)), 7, y);
                    y += 12;
                }
                ctx.restore();
            } catch (e) { /* never break the canvas */ }
        };
    },
};

app.registerExtension(extension);
