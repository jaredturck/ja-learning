# Nihongo Loop

A one-page Japanese sentence construction trainer built with React, TypeScript, Tailwind CSS, and Vite.

## Run locally

```bash
npm install
npm run dev
```

Open the local URL printed by Vite.

## Production build

```bash
npm run build
npm run preview
```

## Add curriculum data

The curriculum lives in `src/levels.ts`. The exported `levels` array is intentionally empty.

Each chunk contains:

- `japanese`: the correct Japanese chunk
- `english`: the direct word or bracketed grammatical idea shown after selection
- `distractors`: the pool used to generate changing Japanese-only wrong answers

The app automatically handles shuffled sentence order, shuffled options, changing distractors, full-sentence resets after a mistake, keyboard controls, level completion, and local progress storage.
