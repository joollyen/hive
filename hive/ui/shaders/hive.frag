#version 440

// HIVE core visualizer — v3.
//
// A blend of the three references:
//   ref 1  a feminine face dissolving into vertical slit-scan light columns,
//          deep electric blue, dark visor band across the eyes
//   ref 2  scrolling terminal glyphs reflected in eyewear, heavy film grain
//   ref 3  a clinical HUD plane over the eye line, white data left, red right
//   ref 4  (v3) a head blurred nearly to a smudge, hair mass carrying the
//          silhouette, crisp instrument text laid over the softness
//
// The face is generated analytically rather than sampled from a texture, so it
// can be deformed per-column by live audio. Quiet: a composed face. Speaking:
// the face tears into vertical streaks whose length tracks the spectrum. The
// dissolve *is* the voice — it is not a meter drawn next to a portrait.
//
// v3 changes three things:
//   - hair. v2 had a haze where hair should be, which read as a bald skull. The
//     mass is now modelled properly (crown, falls, part, shine) and it is also
//     what the outer columns smear into.
//   - a real blur, not a haze. Seven taps on a jittered ring, radius growing
//     outward from the eye line, so the subject is soft the way ref 4 is soft
//     while the glyph band stays crisp and legible.
//   - feminine proportions: narrower jaw, arched brow, almond eyes with lifted
//     outer corners, fuller lips over a defined mouth line, a neck.

layout(location = 0) in vec2 qt_TexCoord0;
layout(location = 0) out vec4 fragColor;

layout(std140, binding = 0) uniform buf {
    mat4  qt_Matrix;
    float qt_Opacity;
    float uTime;        // seconds since start
    float uLevel;       // broadband amplitude 0..1
    float uAlert;       // 0 normal, 1 veto/error (red)
    float uSpeaking;    // 1 while HIVE is talking
    float uThinking;    // 1 while the model is working
    float uDormant;     // 1 when not addressed (dimmed, slow drift)
    float uAspect;      // width / height
    // 64 spectrum bands as 16 vec4s. Separate uniforms rather than an array
    // because ShaderEffect binds QML properties by name, one per uniform.
    vec4  uS0;  vec4 uS1;  vec4 uS2;  vec4 uS3;
    vec4  uS4;  vec4 uS5;  vec4 uS6;  vec4 uS7;
    vec4  uS8;  vec4 uS9;  vec4 uS10; vec4 uS11;
    vec4  uS12; vec4 uS13; vec4 uS14; vec4 uS15;
};

const float PI  = 3.14159265;
const float TAU = 6.28318531;

float hash11(float p) {
    p = fract(p * 0.1031);
    p *= p + 33.33;
    return fract((p + p) * p);
}

float hash21(vec2 p) {
    vec3 p3 = fract(vec3(p.xyx) * 0.1031);
    p3 += dot(p3, p3.yzx + 33.33);
    return fract((p3.x + p3.y) * p3.z);
}

float noise1(float x) {
    float i = floor(x), f = fract(x);
    f = f * f * (3.0 - 2.0 * f);
    return mix(hash11(i), hash11(i + 1.0), f);
}

float noise2(vec2 p) {
    vec2 i = floor(p), f = fract(p);
    f = f * f * (3.0 - 2.0 * f);
    float a = hash21(i);
    float b = hash21(i + vec2(1.0, 0.0));
    float c = hash21(i + vec2(0.0, 1.0));
    float d = hash21(i + vec2(1.0, 1.0));
    return mix(mix(a, b, f.x), mix(c, d, f.x), f.y);
}

// Fractal noise, four octaves. Drives the liquid warp — the melted, flowing
// quality of reference 1 — and the idle drift that keeps the face alive when
// nothing is being said to it.
float fbm(vec2 p) {
    float v = 0.0, amp = 0.5;
    v += amp * noise2(p);       p *= 2.03; amp *= 0.5;
    v += amp * noise2(p);       p *= 2.01; amp *= 0.5;
    v += amp * noise2(p);       p *= 2.05; amp *= 0.5;
    v += amp * noise2(p);
    return v;
}

// Spectrum lookup, 64 bands across 16 vec4 uniforms.
//
// Written entirely in floats with no arrays and no bitwise operators, because
// Qt selects the GLSL ES 1.00 variant of this shader under Wayland/EGL. That
// profile has no array constructors, no `>>`/`&`, and no implicit int-float
// conversion. A branch ladder compiles everywhere and the GPU flattens it.
float band(float fi) {
    float i = clamp(fi, 0.0, 63.0);
    float v = floor(i * 0.25);
    float c = i - v * 4.0;

    vec4 s;
    if      (v < 0.5)  s = uS0;
    else if (v < 1.5)  s = uS1;
    else if (v < 2.5)  s = uS2;
    else if (v < 3.5)  s = uS3;
    else if (v < 4.5)  s = uS4;
    else if (v < 5.5)  s = uS5;
    else if (v < 6.5)  s = uS6;
    else if (v < 7.5)  s = uS7;
    else if (v < 8.5)  s = uS8;
    else if (v < 9.5)  s = uS9;
    else if (v < 10.5) s = uS10;
    else if (v < 11.5) s = uS11;
    else if (v < 12.5) s = uS12;
    else if (v < 13.5) s = uS13;
    else if (v < 14.5) s = uS14;
    else               s = uS15;

    if      (c < 0.5) return s.x;
    else if (c < 1.5) return s.y;
    else if (c < 2.5) return s.z;
    return s.w;
}

// Smoothly interpolated spectrum so 64 bands drive ~96 columns without stepping.
float spectrumAt(float x) {
    float f = clamp(x, 0.0, 1.0) * 63.0;
    float i = floor(f);
    return mix(band(i), band(i + 1.0), f - i);
}

// ---------------------------------------------------------------------------
// The subject. Analytic, built from soft gaussian lobes over an elliptical
// skull, with a hair mass on top of it.
//
// COORDINATES. Texture coordinates run y-down, but describing anatomy y-down is
// how you end up with the chin above the brow. So everything below works in a
// face frame: origin at the eye line, +y toward the crown. faceFrame() is the
// only place the flip happens, and the eye line is shared with the visor so the
// two can never drift apart.
// ---------------------------------------------------------------------------
const float EYE_Y = -0.020;     // p-space row of the eyes and the visor band
// How much of the frame the subject fills. Larger value, smaller head. v2 sat at
// 1.0 and put the mouth behind the log readout at the bottom of the window; with
// hair added on top of that the subject overflowed the frame entirely.
const float SUBJECT = 1.12;

vec2 faceFrame(vec2 pp) {
    return vec2(pp.x, -(pp.y - EYE_Y)) * SUBJECT;
}

float faceField(vec2 f) {
    // Skull, centred a little above the eye line, narrowing toward the jaw.
    //
    // The taper is stronger than v2's and starts higher. A weak taper gives a
    // broad, heavy jaw that reads as male at best and as a skull at worst; jaw
    // width is the single loudest gender cue on a face this abstract.
    float taper = 1.0 - 0.44 * smoothstep(0.03, -0.30, f.y);
    vec2  c     = f - vec2(0.0, 0.045);
    vec2  e     = vec2(c.x / (0.222 * taper), c.y / 0.328);
    // A defined edge rather than a haze — the wide falloff was what made this
    // read as a glowing dome instead of a head.
    float head  = smoothstep(1.03, 0.80, length(e));

    // Broad shading: forehead catches light, jaw falls into shadow.
    //
    // Kept deliberately shallow. A wide base gradient eats the dynamic range the
    // features need — with a 0.46 sweep here the brow, cheek and nose separated
    // by only a few percent of luminance and the whole thing rendered as a smooth
    // glowing oval. The base sets the mass; the features carry the likeness.
    float lift = 0.34 + 0.15 * smoothstep(-0.34, 0.26, f.y);

    // Cheekbones: higher and further out than v2, which is what reads as
    // structure rather than softness. Tighter than the first v3 pass, where
    // sigmas of 0.070 and 0.080 rendered as two headlamps.
    float cheeks = exp(-pow((abs(f.x) - 0.128) / 0.054, 2.0))
                 * exp(-pow((f.y + 0.046) / 0.062, 2.0));
    // Nose: a narrow highlight from the brow to the tip. Kept short and faint —
    // a long bright ridge dominates the whole face, and a wide one masculinises it.
    float nose   = exp(-pow(f.x / 0.017, 2.0))
                 * exp(-pow((f.y + 0.060) / 0.072, 2.0));
    // Brow ridge, arched: the peak sits outboard of centre and falls away again
    // toward the temple. A flat brow line is the other half of a male read.
    float browY  = 0.098 + 0.028 * smoothstep(0.02, 0.125, abs(f.x))
                         - 0.042 * smoothstep(0.145, 0.225, abs(f.x));
    float brow   = exp(-pow((abs(f.x) - 0.110) / 0.052, 2.0))
                 * exp(-pow((f.y - browY) / 0.032, 2.0));

    // Eye sockets. Subtractive, and the single strongest cue that this is a face.
    // Sheared so the outer corner of each rides up — an almond, not a hole.
    vec2 dl = f - vec2(-0.108, 0.004);
    vec2 dr = f - vec2( 0.108, 0.004);
    dl.y += 0.17 * dl.x;
    dr.y -= 0.17 * dr.x;
    vec2 eL = dl / vec2(0.078, 0.029);
    vec2 eR = dr / vec2(0.078, 0.029);
    float eyes = exp(-dot(eL, eL)) + exp(-dot(eR, eR));
    // Lash line: a thin dark stroke along the top of each socket, extended
    // outward. Almost invisible at this blur, and the face is noticeably less
    // doll-like with it than without.
    float lash = exp(-pow((abs(f.x) - 0.112) / 0.070, 2.0))
               * exp(-pow((f.y - 0.030) / 0.008, 2.0));

    // Lips: two lobes with a cupid's bow notch in the upper one, and the mouth
    // line between them. A single lobe reads as a smudge once blurred.
    float bow   = 0.011 * exp(-pow(f.x / 0.024, 2.0));
    vec2  lpU   = (f - vec2(0.0, -0.181 - bow)) / vec2(0.072, 0.016);
    vec2  lpL   = (f - vec2(0.0, -0.222))       / vec2(0.064, 0.020);
    float lips  = 0.82 * exp(-dot(lpU, lpU)) + exp(-dot(lpL, lpL));
    float mouth = exp(-pow((f.y + 0.202) / 0.006, 2.0))
                * exp(-pow(f.x / 0.060, 2.0));
    // The shadow above the lip that separates lip from nose.
    float philtrum = exp(-pow(f.x / 0.026, 2.0))
                   * exp(-pow((f.y + 0.148) / 0.017, 2.0));
    // Chin highlight, so the jaw ends in a point rather than a smudge.
    float chin = exp(-pow(f.x / 0.046, 2.0))
               * exp(-pow((f.y + 0.262) / 0.034, 2.0));

    float v = head * (lift + 0.20 * cheeks + 0.16 * nose + 0.09 * brow + 0.10 * chin);
    v -= 0.70 * eyes * head;
    v -= 0.30 * lash * head;
    v -= 0.20 * philtrum * head;
    v -= 0.24 * mouth * head;
    v += 0.22 * lips * head;
    return clamp(v, 0.0, 1.0);
}

// ---------------------------------------------------------------------------
// Hair.
//
// v2 had a soft halo standing in for hair, which at any blur level reads as a
// bald head with a glow behind it. This is a real mass in three parts: a crown
// cap larger than the skull, two falls that widen as they descend past the jaw,
// and a shine band across the crown. It is also the material the slit-scan wants
// — the outer columns smear hair into the long vertical light streaks that carry
// reference 1, and there was nothing out there to smear before.
// ---------------------------------------------------------------------------
float hairMass(vec2 f) {
    // Crown: an ellipse a little larger than the skull, sitting slightly higher.
    vec2  hc  = (f - vec2(0.0, 0.058)) / vec2(0.294, 0.392);
    float cap = smoothstep(1.06, 0.78, length(hc));

    // Cut the face out of it — but only below the hairline, so the crown stays
    // covered and the forehead does not.
    vec2  fo       = (f - vec2(0.0, 0.020)) / vec2(0.212, 0.320);
    float faceOval = smoothstep(1.00, 0.84, length(fo));
    // A wide transition. A tight one puts a visible edge across the forehead,
    // and a hairline is the last place a hard line belongs on a face this soft.
    float below    = smoothstep(0.285, 0.118, f.y);
    cap *= 1.0 - faceOval * below;

    // Falls: from the temples down past the jaw, widening as they go. Hair that
    // keeps a constant width reads as a helmet.
    float w     = 0.248 + 0.098 * smoothstep(0.14, -0.48, f.y);
    float outer = smoothstep(w + 0.060, w - 0.020, abs(f.x));
    float clear = smoothstep(0.135, 0.195, abs(f.x));   // keep the face uncovered
    float top   = smoothstep(0.36, 0.19, f.y);
    float bot   = smoothstep(-0.72, -0.44, f.y);
    float fall  = outer * clear * top * bot;

    // A side part, off centre. Without it the crown is a moulded shell.
    float part = exp(-pow((f.x + 0.046) / 0.025, 2.0)) * smoothstep(0.18, 0.36, f.y);

    return clamp(max(cap, fall) - 0.55 * part, 0.0, 1.0);
}

float hairShade(vec2 f, float t) {
    // Strands running with the fall of the hair, curving outward as they descend.
    // Noise on its own reads as static; the sine carrier is what makes it hair.
    float s       = f.x * 15.0 + f.y * 2.6 + 1.4 * sin(f.y * 3.1 + f.x * 5.0);
    float strands = 0.5 + 0.5 * sin(s * PI);
    strands = mix(strands, noise2(vec2(f.x * 30.0, f.y * 9.0 - t * 0.05)), 0.42);

    // Shine band across the crown — at this blur it is the cue that separates
    // "hair" from "hood". Curved, not level: a highlight on a round head follows
    // the skull and drops away at the temples, and a straight bar of light across
    // the top of the head reads as a headband.
    float shineY = 0.252 - 1.15 * f.x * f.x;
    float shine  = exp(-pow((f.y - shineY) / 0.058, 2.0))
                 * smoothstep(0.32, 0.04, abs(f.x));

    // Dark base. Hair has to sit well under skin luminance or the silhouette
    // stops separating from the face — but not so far under that it disappears
    // into the background, which is the other way to end up looking bald.
    return 0.30 + 0.10 * strands + 0.13 * shine;
}

// Neck and shoulders — gives the head something to sit on, and more material for
// the outer columns to tear into.
float bodyField(vec2 f) {
    float neck = smoothstep(-0.23, -0.33, f.y) * smoothstep(0.118, 0.068, abs(f.x));
    float sh   = smoothstep(-0.44, -0.62, f.y) * smoothstep(0.68, 0.24, abs(f.x));
    return clamp(neck * 0.28 + sh * 0.32, 0.0, 1.0);
}

// The whole subject, one sample.
float agentField(vec2 f, float t) {
    float m    = hairMass(f);
    float skin = faceField(f) * (1.0 - 0.88 * m);   // hair occludes
    // Rim along the hair's outer boundary. The mass is soft-edged by design, so
    // without this the silhouette dissolves into the background and the head has
    // no outline at all — which at this blur is most of what says "hair" rather
    // than "glow". m*(1-m) peaks exactly on the falloff, and costs nothing.
    float rim  = m * (1.0 - m) * 4.0;
    float hair = (hairShade(f, t) + 0.20 * rim) * m;
    float body = bodyField(f) * (1.0 - 0.70 * m);
    return clamp(max(max(skin, hair), body), 0.0, 1.0);
}

// Soft focus.
//
// Reference 4 is a head blurred nearly to a smudge, and a haze does not
// reproduce that — a haze lowers contrast while leaving every edge exactly where
// it was. This is a real blur: six taps on a ring plus the centre, the ring
// rotated by a per-pixel hash so the kernel's six-fold symmetry never shows as a
// pattern. The rotation is spatial and not temporal, so it does not shimmer.
float softField(vec2 f, float r, float t, float jit) {
    if (r < 0.0015) {
        return agentField(f, t);
    }
    float a  = jit * TAU;
    vec2  d0 = vec2(cos(a), sin(a)) * r;
    vec2  d1 = vec2(cos(a + 2.0944), sin(a + 2.0944)) * r;
    vec2  d2 = vec2(cos(a + 4.1888), sin(a + 4.1888)) * r;
    float acc = agentField(f, t) * 1.30;
    acc += agentField(f + d0, t) + agentField(f - d0, t);
    acc += agentField(f + d1, t) + agentField(f - d1, t);
    acc += agentField(f + d2, t) + agentField(f - d2, t);
    return acc / 7.30;
}

// Cheaper horizontal-only blur for the slit-scanned sample. That sample is
// already smeared along y by the scan, so blurring it along y as well buys
// nothing; across x is the axis that still carries edges.
float softFieldH(vec2 f, float r, float t) {
    return (agentField(f, t) * 1.20
          + agentField(f + vec2(r, 0.0), t)
          + agentField(f - vec2(r, 0.0), t)) / 3.20;
}

// ---------------------------------------------------------------------------
// Scrolling glyph field for the visor. Procedural monospace-ish blocks.
//
// Deliberately NOT blurred. The whole composition rests on the contrast between
// a subject soft enough to be a memory and instrument text sharp enough to read
// — that is what keeps this a coding agent rather than a portrait.
// ---------------------------------------------------------------------------
float glyphField(vec2 uv, float t, float density) {
    vec2 g    = vec2(uv.x * 104.0 + t * 7.0, uv.y * 300.0);
    vec2 cell = floor(g);
    vec2 f    = fract(g);
    // Character presence, reshuffled in blocks so it reads as words not static.
    float word = hash21(vec2(floor(cell.x / 4.0), cell.y) + floor(t * 1.7) * 13.0);
    float chr  = hash21(cell + floor(t * 5.0) * 31.0);
    float on   = step(1.0 - density, word * 0.55 + chr * 0.45);
    // Glyph body with inter-character gaps.
    float m = step(0.14, f.x) * step(f.x, 0.84) * step(0.20, f.y) * step(f.y, 0.80);
    return on * m;
}

// Palette moved toward the teal-cyan of references 2 and 3 — the bioluminescent
// jellyfish and the neural-activity scan — away from v1's flatter electric blue.
vec3 palette(float v) {
    // Brighter through the middle than the first teal attempt, which was so dark
    // the face read as a faint silhouette rather than something present. The
    // ramp has to carry the subject; the network is only atmosphere around it.
    // An S-curve first. Without it the face renders as a smooth glowing oval:
    // the shading spans roughly 0.34-1.0, and a linear ramp puts almost all of
    // that in the bright half where the brow, cheek and nose separate by only a
    // few percent of luminance. Pushing the mids down is what makes the features
    // legible at all — and it matters more, not less, once the whole subject is
    // blurred, because blur is itself a contrast cut.
    v = clamp(v, 0.0, 1.0);
    v = mix(v, v * v * (3.0 - 2.0 * v), 0.55);

    const vec3 deep = vec3(0.006, 0.022, 0.034);
    const vec3 teal = vec3(0.018, 0.185, 0.210);
    const vec3 cyan = vec3(0.230, 0.850, 0.870);
    const vec3 pale = vec3(0.870, 1.000, 0.995);
    vec3 c = mix(deep, teal, smoothstep(0.00, 0.38, v));
    c = mix(c, cyan, smoothstep(0.34, 0.76, v));
    c = mix(c, pale, smoothstep(0.72, 1.00, v));
    return c;
}

// ---------------------------------------------------------------------------
// Node-and-filament network, from reference 3's brain scan: a sparse lattice of
// glowing points joined by faint threads, with a few hot ones in orange.
//
// One node per grid cell, searched across the 3x3 neighbourhood — so the cost is
// nine distance tests per pixel rather than N-squared over a node list.
//
// v3 pushes this much further out of focus than v2. It is background, and the
// brief was a background well behind the subject: fewer cells, far wider dots,
// soft threads, roughly half the intensity.
// ---------------------------------------------------------------------------
vec3 network(vec2 uv, float t, float level, float alert) {
    const float CELLS = 5.0;
    vec2 gp = uv * CELLS;
    vec2 cell = floor(gp);

    vec3 acc = vec3(0.0);
    for (float oy = -1.0; oy <= 1.0; oy += 1.0) {
        for (float ox = -1.0; ox <= 1.0; ox += 1.0) {
            vec2 c = cell + vec2(ox, oy);
            float h = hash21(c);
            // Each node drifts on its own slow orbit; nothing sits still.
            vec2 node = c + 0.5 + 0.34 * vec2(sin(t * (0.20 + h * 0.28) + h * 31.0),
                                              cos(t * (0.17 + h * 0.24) + h * 17.0));
            float d = length(gp - node);

            float pulse = 0.55 + 0.45 * sin(t * (0.7 + h) + h * 12.0);
            // 42 -> 9: the dots become soft blooms rather than points.
            float dot_ = exp(-d * d * 20.0) * (0.35 + 0.65 * pulse);
            // A minority run hot, which is what gives reference 3 its red foci.
            float hot = step(0.86, h);
            vec3 col = mix(vec3(0.22, 0.88, 0.86), vec3(1.00, 0.42, 0.18), hot);
            col = mix(col, vec3(1.0, 0.25, 0.22), alert * 0.8);
            acc += col * dot_;

            // Filament toward the home cell's node.
            float hc = hash21(cell);
            vec2 home = cell + 0.5 + 0.34 * vec2(sin(t * (0.20 + hc * 0.28) + hc * 31.0),
                                                 cos(t * (0.17 + hc * 0.24) + hc * 17.0));
            vec2 ab = node - home;
            float len2 = max(dot(ab, ab), 1e-5);
            float proj = clamp(dot(gp - home, ab) / len2, 0.0, 1.0);
            float dist = length(gp - (home + ab * proj));
            float thread = exp(-dist * dist * 320.0) * step(length(ab), 1.5);
            acc += col * thread * 0.12;
        }
    }
    return acc * (0.07 + 0.13 * level);
}

void main() {
    vec2 uv = qt_TexCoord0;
    vec2 p  = (uv - 0.5) * vec2(uAspect, 1.0);

    float t     = uTime;
    float sleep = uDormant;
    // Dormant: quieter, slower, dimmer. Awake: responsive.
    float react = mix(1.0, 0.30, sleep);

    // -- autonomous life ------------------------------------------------------
    // v1 sat perfectly still until spoken to, which read as a screensaver rather
    // than something present. Three layers of motion, none of them driven by
    // audio, so the face is alive in silence:
    //
    //   sway    a slow lissajous, like weight shifting from foot to foot
    //   breath  a ~0.1Hz scale pulse
    //   warp    fbm domain distortion — reference 1's liquid, melted quality
    //
    // The periods are deliberately incommensurate (0.23, 0.19, 0.41, 0.37) so the
    // combined motion never visibly repeats. Anything that loops on a countable
    // beat stops looking alive within about a minute of watching it.
    float breath = 0.5 + 0.5 * sin(t * 0.62);
    float slowB  = 0.5 + 0.5 * sin(t * 0.21 + 0.8);

    float swayX = 0.017 * sin(t * 0.23) + 0.008 * sin(t * 0.41 + 1.7);
    float swayY = 0.011 * sin(t * 0.19 + 2.1) + 0.005 * sin(t * 0.37);
    swayX += 0.007 * (fbm(vec2(t * 0.12, 3.7)) - 0.5);
    swayY += 0.007 * (fbm(vec2(9.1, t * 0.11)) - 0.5);
    // Dormant still moves, just less. Freezing it entirely is the thing we are
    // fixing, so the floor is 45% rather than zero.
    float motion = mix(1.0, 0.45, sleep);
    swayX *= motion;
    swayY *= motion;

    // Low frequency and small amplitude, deliberately. Large-scale flow displaces
    // the face; fine-scale noise dissolves it, and reference 1's liquid quality is
    // the former. The soft-focus pass below is where softness comes from now, so
    // this stays a displacement and does not try to double as a blur.
    float warpAmt = (0.004 + 0.010 * uLevel + 0.003 * breath) * motion;
    vec2 warp = vec2(fbm(p * 1.6 + vec2(t * 0.13, 0.0)) - 0.5,
                     fbm(p * 1.8 + vec2(0.0, t * 0.11)) - 0.5) * warpAmt;

    float scale = 1.0 - 0.016 * breath * motion;
    p = (p - vec2(swayX, swayY)) / scale + warp;

    // -- column quantisation -------------------------------------------------
    const float COLS = 96.0;
    float colId  = floor(uv.x * COLS);
    float colCtr = (colId + 0.5) / COLS;

    // Per-column audio energy, plus a slow idle breathing so it never flatlines.
    float energy = spectrumAt(colCtr);
    // Idle floor breathes with the body, so the columns keep moving in silence.
    float idle   = 0.13 + 0.09 * noise1(colId * 0.19 + t * 0.55)
                 + 0.05 * sin(t * 0.5 + colId * 0.11) * slowB;
    energy = max(energy * react, idle * mix(0.55, 1.0, sleep));

    // -- slit-scan ------------------------------------------------------------
    // Each column freezes toward a single sample row. At smear=0 the face is
    // intact; at smear=1 the column becomes one solid vertical streak. Driving
    // smear from the spectrum makes the face tear apart exactly on the voice.
    float jitter  = noise1(colId * 0.37 + t * 0.9);
    float smear   = clamp(energy * 0.92 + uLevel * 0.45 * react, 0.0, 1.0);
    smear = pow(smear, 1.35) * mix(0.86, 0.30, sleep);
    // Shred the silhouette. Columns away from centre smear much harder, so the
    // outline tears into vertical light streaks while the features stay legible
    // in the middle — the defining move of reference 1.
    float edgeBias = smoothstep(0.12, 0.34, abs(p.x));
    smear = clamp(smear + edgeBias * (0.34 + 0.46 * energy), 0.0, 1.0);
    // Cap the centre hard. Smear is a mix weight toward a frozen sample row, so
    // even 0.7 in the middle means the face is mostly gone and the result reads
    // as a bar chart. Reference 1 keeps the features plainly readable and puts
    // the long streaks in the hair and outline — so the core stays under 0.4 and
    // only the outer columns are allowed to tear all the way.
    smear = min(smear, mix(0.38, 1.0, edgeBias));

    // Anchor row this column smears toward, drifting slowly.
    float anchor  = 0.50 + 0.30 * sin(colId * 0.21 + t * 0.42) * (0.35 + 0.65 * jitter);
    float sampleY = mix(uv.y, anchor, smear);
    // Vertical slide, so streaks travel rather than sitting still.
    sampleY += (jitter - 0.5) * smear * 0.16;

    // -- the subject, softly ---------------------------------------------------
    // Sampled twice — once undisplaced, once slit-scanned — and blended.
    //
    // Displacing a single sample means a loud passage displaces every column at
    // once and the face disappears completely. Keeping the clean sample in the
    // mix guarantees the features survive no matter how hot the spectrum gets;
    // the dissolve then rides on top, weighted outward, so the hair and outline
    // tear into streaks while the eyes and mouth stay readable.
    vec2 fClean = faceFrame(p);
    vec2 fs     = faceFrame(vec2(p.x, sampleY - 0.5));

    // Blur radius grows outward from the eye line. The band across the eyes and
    // the mouth keep enough definition to read as features; the jaw, the hair and
    // the shoulders go soft, which is the gradient reference 4 has.
    // The radii are smaller than they look. The finest feature on this face —
    // the nose highlight — is 0.017 wide, so a 0.05 kernel does not soften it,
    // it deletes it, and the first version of this blurred the entire subject
    // out of existence. Softness comes from the kernel being *comparable* to the
    // features, not larger than them.
    float dEye = length(vec2(fClean.x * 0.88, fClean.y - 0.02));
    float blur = 0.008 + 0.030 * smoothstep(0.09, 0.50, dEye) + 0.008 * uLevel;
    blur *= mix(1.0, 1.25, sleep);            // dormant is dreamier still
    // Spatial, not temporal: a per-frame hash here makes the blur boil.
    float jit = hash21(floor(uv * 271.0));

    float vClean = softField(fClean, blur, t, jit);
    float vSmear = softFieldH(fs, blur * 0.8, t);

    float dissolve = clamp(edgeBias * 0.80 + smear * 0.40, 0.0, 1.0);
    float v = mix(vClean, vSmear, dissolve);

    // Blur is a contrast cut. Averaging seven taps over a field whose features
    // are about as wide as the kernel pulls everything toward the middle of the
    // range, and the palette's S-curve — which is tuned to push mids *down* so
    // the brow and cheek separate — then crushes what survived to black. The
    // range has to be put back before the ramp sees it, not inside the ramp,
    // or the curve stops doing the job it was written for.
    // A power curve, not a linear stretch.
    //
    // A linear expansion put the face's whole tonal range inside the palette's
    // steepest segment, where teal runs to cyan across 0.42 of the input. Every
    // feature then landed a stop brighter than the mass it sits on, and the
    // result was a glowing mask floating in the dark rather than a head. This
    // pulls the mids down while pinning zero at zero — the background has to
    // stay black, so an additive offset is not available — which puts skin in
    // the teal band, hair below it, and leaves only genuine highlights in cyan.
    v = pow(clamp(v, 0.0, 1.0), 1.30) * 1.15;

    // Columns read as discrete bands of light: brighten the ones with energy,
    // and let the gutters between them stay dark. The boost is modest on
    // purpose — pushed harder, the outer columns outshine the face.
    float gutter = abs(fract(uv.x * COLS) - 0.5) * 2.0;
    float bandLight = smoothstep(0.98, 0.10, gutter);
    v *= mix(1.0, 0.62 + 0.62 * energy, dissolve * 0.85);
    v *= mix(1.0, bandLight, 0.14 + 0.40 * dissolve);

    // A little vertical streak texture inside heavily-smeared columns.
    v += dissolve * 0.14 * noise1(uv.y * 90.0 + colId * 3.1 + t * 1.2) * step(0.25, dissolve);

    // -- visor ---------------------------------------------------------------
    float visorC = EYE_Y;                 // shares the eye line with the face
    float visorH = 0.050 + 0.012 * uThinking;
    float dv     = abs(p.y - visorC);
    float visor  = smoothstep(visorH, visorH * 0.50, dv);
    float visorEdge = smoothstep(visorH * 1.06, visorH * 0.92, dv)
                    - smoothstep(visorH * 0.92, visorH * 0.70, dv);

    // Crush the band, then lay glyphs over it (ref 1 + ref 2). Not all the way to
    // black any more: at 0.06 the eyes vanished entirely and the band read as a
    // blindfold. At 0.14 they ghost through it, which is the reference.
    v = mix(v, v * 0.14, visor);

    float density = 0.30 + 0.34 * uLevel + 0.22 * uThinking + 0.14 * uSpeaking;
    float gl = glyphField(vec2(uv.x, uv.y), t, clamp(density, 0.0, 0.92)) * visor;

    // -- colour --------------------------------------------------------------
    vec3 col = palette(clamp(v, 0.0, 1.0));

    // Glyphs: white-cyan on the left, warm/red on the right — ref 3's split HUD.
    vec3 glyphCool = vec3(0.72, 0.94, 1.00);
    vec3 glyphWarm = vec3(1.00, 0.35, 0.32);
    float split = smoothstep(0.54, 0.80, uv.x);
    vec3 glyphCol = mix(glyphCool, glyphWarm, max(split * 0.85, uAlert));
    col += gl * glyphCol * (0.55 + 0.75 * uLevel);

    // Node-and-filament network over the whole frame — reference 3's neural
    // activity map, pushed well out of focus in v3.
    vec3 net = network(uv, t, 0.25 + 0.75 * uLevel, uAlert);
    // Fades out over the subject. The network frames it; letting it sit on top
    // made the two compete and the face lose.
    net *= (1.0 - visor * 0.85) * (1.0 - 0.80 * smoothstep(0.14, 0.55, v));
    col += net * (0.55 + 0.45 * slowB);

    // A wide, very soft glow behind the head. Reference 4's subject sits in a
    // pool of light rather than on flat black, and that pool is most of why the
    // blur reads as depth instead of as a smudge.
    float halo = exp(-pow(length(vec2(p.x * 1.15, (p.y + 0.02) * 0.92)) / 0.40, 2.0));
    col += vec3(0.030, 0.115, 0.130) * halo * (0.45 + 0.22 * uLevel + 0.25 * slowB);

    // Visor rim highlight.
    col += visorEdge * vec3(0.35, 0.72, 1.0) * (0.30 + 0.55 * uLevel);

    // Thinking: a scan sweep travelling down the frame.
    float sweep = exp(-pow((uv.y - fract(t * 0.33)) * 9.0, 2.0)) * uThinking;
    col += sweep * vec3(0.18, 0.44, 0.78) * 0.55;

    // Alert: bias the whole frame red without losing the structure.
    col = mix(col, vec3(col.r * 1.55 + 0.10, col.g * 0.28, col.b * 0.30), uAlert * 0.80);

    // -- post ----------------------------------------------------------------
    // Chromatic aberration, strongest at the edges (ref 2's lens feel). Taken on
    // the unblurred skin field: run through the blurred composite it cancels out.
    float ca = 0.0016 + 0.0042 * uLevel;
    float faceRaw = faceField(fs);
    float rShift  = faceField(vec2(fs.x - ca, fs.y));
    float bShift  = faceField(vec2(fs.x + ca, fs.y));
    col.r += (rShift - faceRaw) * 0.30;
    col.b += (bShift - faceRaw) * 0.30;

    // Scanlines, softer than v2's. Hard scanlines fight a soft subject — they
    // reintroduce exactly the high-frequency edge the blur just removed.
    col *= 1.0 - 0.055 * (0.5 + 0.5 * sin(uv.y * 620.0));
    // Film grain — ref 2 is heavily grained and it does a lot of the work. More
    // of it in v3: grain is what stops a heavy blur reading as an out-of-focus
    // photograph rather than a rendered presence.
    float grain = hash21(uv * vec2(1024.0, 768.0) + fract(t) * 91.7) - 0.5;
    col += grain * (0.070 + 0.030 * uLevel);
    // Vignette, tighter than v2's.
    float vig = smoothstep(1.04, 0.26, length((uv - 0.5) * vec2(uAspect, 1.15)));
    col *= 0.30 + 0.70 * vig;

    // Dormant: desaturate and dim, so waking is visibly a state change.
    float lum = dot(col, vec3(0.299, 0.587, 0.114));
    col = mix(col, vec3(lum) * vec3(0.62, 0.80, 1.0), sleep * 0.42);
    col *= mix(1.0, 0.74, sleep);

    fragColor = vec4(clamp(col, 0.0, 1.0), 1.0) * qt_Opacity;
}
