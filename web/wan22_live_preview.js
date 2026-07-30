const { app } = window.comfyAPI.app;
const { api } = window.comfyAPI.api;

const EVENT_NAME = "wan22_live_preview";
const NODE_NAME = "Wan22LivePreview";
const MIN_NODE_WIDTH = 360;
const MIN_NODE_HEIGHT = 330;

function nodeIdCandidates(value) {
    const candidates = [value, String(value)];
    const numeric = Number(value);
    if (Number.isFinite(numeric)) candidates.push(numeric);
    return [...new Set(candidates)];
}

function findNodeInGraph(graph, id) {
    for (const candidate of nodeIdCandidates(id)) {
        const node = graph?.getNodeById?.(candidate);
        if (node) return node;
    }
    return null;
}

// Execution IDs inside subgraphs are qualified as "parent:child[:leaf]".
function findNodeByQualifiedId(rootGraph, qualifiedId) {
    if (!rootGraph || qualifiedId == null) return null;

    const parts = String(qualifiedId).split(":");
    let graph = rootGraph;
    for (let index = 0; index < parts.length - 1; index += 1) {
        const parent = findNodeInGraph(graph, parts[index]);
        if (!parent?.subgraph) return null;
        graph = parent.subgraph;
    }
    return findNodeInGraph(graph, parts.at(-1));
}

function finiteNumber(...values) {
    for (const value of values) {
        if (value === null || value === undefined || value === "") continue;
        const numeric = Number(value);
        if (Number.isFinite(numeric)) return numeric;
    }
    return null;
}

function formatNumber(value, digits = 3) {
    if (!Number.isFinite(value)) return "—";
    return value.toLocaleString(undefined, {
        maximumFractionDigits: digits,
        useGrouping: false,
    });
}

function formatSigma(value) {
    if (!Number.isFinite(value)) return "—";
    if (value === 0) return "0";
    if (Math.abs(value) < 0.001) return value.toExponential(2);
    return formatNumber(value, 5);
}

function normaliseStage(value) {
    if (value == null || value === "") return "SAMPLING";
    const stage = String(value).replaceAll("_", " ").trim();
    return stage.toUpperCase();
}

function statusText(data) {
    const globalStep = finiteNumber(data.global_step, data.step);
    const totalSteps = finiteNumber(
        data.global_total,
        data.total_steps,
        data.total,
    );
    const sigma = finiteNumber(data.sigma);
    const frameCount = finiteNumber(data.frame_count, data.frames);
    const decodedFrameCount = finiteNumber(data.decoded_frame_count);
    const outputFps = finiteNumber(data.output_fps, data.fps);
    const latentFps = finiteNumber(
        data.effective_latent_fps,
        data.latent_fps,
        outputFps !== null ? outputFps / 4 : null,
    );

    const step = globalStep === null ? "—" : formatNumber(globalStep, 0);
    const total = totalSteps === null ? "—" : formatNumber(totalSteps, 0);
    const frames = frameCount === null ? "—" : formatNumber(frameCount, 0);
    const decodedFrames = decodedFrameCount === null
        ? "—"
        : formatNumber(decodedFrameCount, 0);

    return [
        normaliseStage(data.stage),
        `global ${step}/${total}`,
        `σ ${formatSigma(sigma)}`,
        `${frames} latent / ${decodedFrames} output frames`,
        `output ${formatNumber(outputFps, 2)} fps`,
        `latent ${formatNumber(latentFps, 2)} fps`,
    ].join("  •  ");
}

function webpBase64(data) {
    const value = data.webp_base64 ?? data.webp ?? data.image ?? data.payload;
    if (typeof value !== "string" || value.length === 0) return null;
    const comma = value.indexOf(",");
    return value.startsWith("data:") && comma >= 0 ? value.slice(comma + 1) : value;
}

function base64ToBlob(base64, mime = "image/webp") {
    const binary = atob(base64.replaceAll(/\s/g, ""));
    const bytes = new Uint8Array(binary.length);
    for (let index = 0; index < binary.length; index += 1) {
        bytes[index] = binary.charCodeAt(index);
    }
    return new Blob([bytes], { type: mime });
}

api.addEventListener(EVENT_NAME, (event) => {
    const data = event?.detail;
    const targetId = data?.node_id ?? data?.display_node_id;
    if (!data || targetId == null) return;

    const node = findNodeByQualifiedId(app.graph, targetId);
    node?._wan22LivePreviewHandler?.(data);
});

app.registerExtension({
    name: "Wan22Combined.LivePreview",

    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData?.name !== NODE_NAME) return;

        const originalOnNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function (...args) {
            const result = originalOnNodeCreated?.apply(this, args);
            const node = this;

            const root = document.createElement("div");
            Object.assign(root.style, {
                boxSizing: "border-box",
                display: "flex",
                flexDirection: "column",
                width: "100%",
                height: "100%",
                minHeight: "250px",
                overflow: "hidden",
                border: "1px solid var(--border-color, #404040)",
                borderRadius: "8px",
                background: "var(--comfy-input-bg, #181818)",
                color: "var(--input-text, #ddd)",
                fontFamily: "system-ui, sans-serif",
            });

            const media = document.createElement("div");
            Object.assign(media.style, {
                position: "relative",
                display: "grid",
                placeItems: "center",
                flex: "1 1 auto",
                minHeight: "190px",
                overflow: "hidden",
                background: "#0b0b0b",
            });

            const image = document.createElement("img");
            image.alt = "Wan 2.2 live latent video preview";
            image.draggable = false;
            Object.assign(image.style, {
                display: "none",
                width: "100%",
                height: "100%",
                objectFit: "contain",
                imageRendering: "auto",
                userSelect: "none",
            });

            const placeholder = document.createElement("div");
            placeholder.textContent = "Waiting for Wan sampling…";
            Object.assign(placeholder.style, {
                padding: "18px",
                color: "var(--descrip-text, #999)",
                fontSize: "13px",
                textAlign: "center",
            });

            const status = document.createElement("div");
            status.setAttribute("role", "status");
            status.setAttribute("aria-live", "polite");
            status.textContent = "No preview received";
            Object.assign(status.style, {
                flex: "0 0 auto",
                boxSizing: "border-box",
                minHeight: "44px",
                padding: "9px 11px",
                borderTop: "1px solid var(--border-color, #404040)",
                color: "var(--descrip-text, #aaa)",
                fontSize: "11px",
                lineHeight: "1.35",
                overflowWrap: "anywhere",
            });

            media.append(image, placeholder);
            root.append(media, status);
            node.addDOMWidget("wan22_live_preview", "wan22_live_preview", root, {
                serialize: false,
            });

            const width = Math.max(node.size?.[0] ?? 0, MIN_NODE_WIDTH);
            const height = Math.max(node.size?.[1] ?? 0, MIN_NODE_HEIGHT);
            node.setSize([width, height]);

            let activeRunId = null;
            let activeUrl = null;
            let pendingUrl = null;
            const retiredRunIds = new Set();

            function revoke(url) {
                if (!url) return;
                try {
                    URL.revokeObjectURL(url);
                } catch {
                    // Object URLs are best-effort browser resources.
                }
            }

            function clearMedia() {
                revoke(pendingUrl);
                revoke(activeUrl);
                pendingUrl = null;
                activeUrl = null;
                image.removeAttribute("src");
                image.style.display = "none";
                placeholder.style.display = "block";
            }

            function acceptRun(data) {
                const runId = data.run_id == null ? null : String(data.run_id);
                if (runId === null) return true;
                if (runId === activeRunId) return true;
                if (retiredRunIds.has(runId)) return false;

                // Comfy runs this node serially. The first event bearing a new run ID
                // becomes authoritative; late encoder messages from an older run are
                // then rejected by the retired-ID set.
                if (activeRunId !== null) {
                    retiredRunIds.add(activeRunId);
                    if (retiredRunIds.size > 64) {
                        retiredRunIds.delete(retiredRunIds.values().next().value);
                    }
                }
                activeRunId = runId;
                clearMedia();
                return true;
            }

            function showWebp(base64, data) {
                let nextUrl;
                try {
                    const mime = String(data.mime ?? "image/webp");
                    nextUrl = URL.createObjectURL(base64ToBlob(base64, mime));
                } catch (error) {
                    console.warn("[Wan22LivePreview] Invalid preview payload:", error);
                    return;
                }

                revoke(pendingUrl);
                pendingUrl = nextUrl;
                const expectedRunId = activeRunId;
                const probe = new Image();

                probe.onload = () => {
                    if (pendingUrl !== nextUrl || activeRunId !== expectedRunId) {
                        revoke(nextUrl);
                        return;
                    }
                    pendingUrl = null;
                    const previousUrl = activeUrl;
                    activeUrl = nextUrl;
                    image.src = nextUrl;
                    image.style.display = "block";
                    placeholder.style.display = "none";
                    revoke(previousUrl);
                };
                probe.onerror = () => {
                    if (pendingUrl === nextUrl) pendingUrl = null;
                    revoke(nextUrl);
                    console.warn("[Wan22LivePreview] Browser could not decode animated WebP.");
                };
                probe.src = nextUrl;
            }

            node._wan22LivePreviewHandler = (data) => {
                if (!acceptRun(data)) return;
                status.textContent = statusText(data);

                const encoded = webpBase64(data);
                if (encoded) showWebp(encoded, data);
            };

            const originalOnRemoved = node.onRemoved;
            node.onRemoved = function (...removedArgs) {
                node._wan22LivePreviewHandler = null;
                clearMedia();
                return originalOnRemoved?.apply(this, removedArgs);
            };

            return result;
        };
    },
});
