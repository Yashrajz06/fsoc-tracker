# What this project is, in plain English

Written for someone with no background in optics, control theory or computer vision.

---

## The problem

Imagine two ships at sea at night, each holding a torch, trying to send messages by flashing
light at each other. Except these aren't ordinary torches — each beam is about as wide as a
pencil. If either ship's aim drifts even slightly, the light misses completely and the message
is lost. And both ships are moving.

That is laser communication. It is used between satellites, aircraft, drones and ground stations
because a laser carries far more data than radio does. The catch is aiming. A laser beam spreads
so little that over a long distance, being off by a hair's breadth means missing entirely.

So before two terminals can exchange any data, they have to **find each other and lock on**. This
happens in two stages:

- **Coarse alignment** — roughly find the other terminal and get it into view
- **Fine alignment** — precisely lock the beam

ISRO's problem statement asks for the **coarse stage**, built entirely in software.

## Why build it in software?

The real hardware is expensive, and testing it is harder still — you would need two moving
platforms and a lot of open space. So the ask is for a **simulator**: a virtual world with a
virtual camera, where the tracking method can be developed, tested and measured before any
hardware exists.

## The setup

The far terminal carries a **beacon** — a bright light, effectively a torch pointed back at you.
Your terminal has a camera on a motorised mount that can swivel left/right and up/down.

The job, repeated thirty times a second:

1. Look at the camera image
2. Find the bright dot
3. Work out exactly where it is — to better than a single pixel
4. Swivel the camera so the dot moves to the centre
5. Keep doing it as both platforms move

Get that working and the fine-pointing stage can take over.

## What makes it hard

**You see very little at a time.** The camera views a narrow slice of the world, like looking
through a drinking straw. Our virtual world is 2000x2000 pixels; the camera sees 640x480 of it.

**The picture is bad.** Sensor noise, atmospheric haze and fog, shake from the moving platform,
and damage from video compression. The beacon can be a faint smudge surrounded by bright specks
of noise that look a lot like it.

**The camera cannot swivel infinitely fast.** At full speed it moves about 27 pixels between
frames. Anything crossing faster than that cannot be kept centred. That is physics, not a bug.

**We do not know what the real test looks like.** Nearly a third of the grade comes from video
files the evaluators supply and we never see in advance. Anything tuned to our own test data is a
trap.

---

## How we solve it

**1. Build a believable world.** Render a scene with a beacon moving along realistic paths —
straight lines, circles, figure-eights, random drift. The simulator knows exactly where the
beacon is at all times, which is what lets us score ourselves honestly later.

**2. Make it realistically bad.** Add sensor noise, fog, haze, camera shake and compression
damage, all adjustable.

**3. Find the dot.** Standard image processing: clean up the image, remove the background, find
the bright blobs. The critical rule we follow throughout: **never hardcode what "bright" means.**
Every threshold is calculated from the picture currently in front of us. A number tuned to our
own images would fail on the evaluator's.

**4. Pin down its exact position.** Take a brightness-weighted average of the pixels in the blob
— essentially its centre of gravity. This gets us accuracy far finer than one pixel.

**5. Predict where it is going.** We use a Kalman filter, a standard technique for tracking
something that moves predictably. It gives two big wins:

   - If the beacon briefly disappears behind noise, we keep tracking on the prediction instead of
     losing it
   - Because we know roughly where it will appear, we only need to search a **small window** of
     the next image rather than all of it — which is what makes the whole thing fast enough to
     run in real time

**6. Steer the camera.** A controller converts "the dot is 30 pixels left of centre" into a
smooth camera movement — fast enough to catch up, damped enough not to overshoot and oscillate.

**7. Measure everything.** How far off we were each frame, how long we took to find the target,
how often we lost it, how fast we ran.

---

## Where is the AI?

The honest answer: **mostly it is not, and that is a deliberate engineering decision.**

Finding a bright dot in an image is something classical mathematics does *better* than a neural
network — more accurately, and hundreds of times faster. Using deep learning for that step would
make the system worse on both counts.

But there is one place where classical logic genuinely has nothing to work with. When the image
is full of noise specks, the system picks the *brightest* blob — and sometimes a noise speck is
brighter than a dim beacon. Brightness alone cannot separate them. **Shape** can: a beacon is a
smooth round glow, a compression artifact is a hard-edged block.

So we built a small neural network whose only job is to judge shape and say which blob looks like
a real beacon. We trained it, measured it against the classical method on equal terms — and it
did not earn its place. It helps on some inputs and makes things clearly worse on others. So it
ships **switched off**.

That is a real result rather than a failure. "We built it, measured it, and the simple method
won" is a stronger engineering answer than never having checked.

---

## Two ways to run it

- **Mode A** — the full simulation, with our virtual camera doing the steering
- **Mode B** — feed in a video file; the video *is* the scene, and there is no steering to do

Both modes run through **identical** detection code. That matters: it means the thing we tested
in simulation is exactly the thing the evaluators' videos will exercise.

---

## Where it stands

It works. It typically finds the beacon, locks on, and holds it — tracking to well under one
pixel of error on most tests, against a requirement of ten.

Two limits we report openly rather than hide, because both are real and neither is fixable by
tuning:

- **If the beacon starts outside the camera's view**, finding it takes around ten seconds, not
  the two the specification asks for. That is arithmetic: sweeping a large area at a limited
  swivel speed simply takes that long.
- **One deliberately brutal test case** — very dim beacon, heavy speckle noise, heavy compression
  — never locks at all. The beacon is genuinely not distinguishable in most of those frames.

Work is ongoing; `docs/HANDOFF.md` records exactly what is settled and what is still open.
