export interface SentenceChunk {
    japanese: string
    english: string
    distractors: string[]
}

export interface QuizSentence {
    id: string
    english: string
    chunks: SentenceChunk[]
}

export interface QuizLevel {
    id: string
    title: string
    description: string
    sentences: QuizSentence[]
}

export const levels: QuizLevel[] = []
