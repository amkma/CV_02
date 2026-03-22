/**
 * sliders.js – Convert number inputs with data-slider="true" into
 *              range slider + value badge combos.
 */
document.addEventListener("DOMContentLoaded", () => {
    document.querySelectorAll('input[data-slider="true"]').forEach(input => {
        const min  = parseFloat(input.min);
        const max  = parseFloat(input.max);
        const step = parseFloat(input.step) || 1;
        const val  = parseFloat(input.value) || min;

        // Create wrapper
        const wrapper = document.createElement("div");
        wrapper.className = "slider-wrap";

        // Range input
        const range = document.createElement("input");
        range.type  = "range";
        range.min   = min;
        range.max   = max;
        range.step  = step;
        range.value = val;
        range.className = "slider-range";

        // Value badge
        const badge = document.createElement("span");
        badge.className = "slider-val";
        badge.textContent = val;

        // Hide original, keep it in the form for submission
        input.type = "hidden";

        // Sync range → hidden + badge
        range.addEventListener("input", () => {
            const v = parseFloat(range.value);
            input.value = v;
            badge.textContent = step < 1 ? v.toFixed(1) : v;
            updateTrackFill(range);
        });

        // Insert after the hidden input
        input.parentNode.insertBefore(wrapper, input.nextSibling);
        wrapper.appendChild(range);
        wrapper.appendChild(badge);

        // Initial track fill
        updateTrackFill(range);
    });

    function updateTrackFill(range) {
        const min = parseFloat(range.min);
        const max = parseFloat(range.max);
        const val = parseFloat(range.value);
        const pct = ((val - min) / (max - min)) * 100;
        range.style.setProperty("--fill", pct + "%");
    }
});
