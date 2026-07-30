import logging

from google.adk.agents import Agent, SequentialAgent
from google.adk.models.google_llm import Gemini

from app.toolsets import retry_config

from app.callbacks import generic_callback, skip_if_extract_only

logger = logging.getLogger(__name__)

# Exam-standard rules, kept verbatim in sync with
# gradr_backend/utils/examBlueprints.js. Selected by the `standard` field of
# the incoming payload. Previously the backend held these strings but never
# sent them anywhere, so the standard selector had no effect on output.
STANDARD_RULES = (
    "<standard_rules>\n"
    "Apply ONLY the block matching the payload's `standard` field.\n\n"
    "[GENERIC]\n"
    "- Generate clear, unambiguous questions.\n"
    "- Ensure the questions are unique and directly derived from the provided context.\n\n"
    "[JAMB]\n"
    "You are an expert JAMB (Joint Admissions and Matriculation Board) exam setter.\n"
    "1. Format: STRICTLY Multiple Choice Questions (MCQ).\n"
    "2. Options: Exactly 4 options (A, B, C, D) per question.\n"
    "3. Style: Concise stems. Focus on speed, critical thinking, and logical deduction.\n"
    "4. No elaborate preambles unless necessary for a set of questions (e.g., comprehension).\n"
    '5. Avoid "All of the above" or "None of the above" unless absolutely necessary.\n'
    "6. Ambiguity: Ensure there is exactly ONE correct answer.\n"
    "7. Language: Use formal British English as used in Nigeria.\n\n"
    "[WASSCE]\n"
    "You are an expert WAEC/WASSCE examiner.\n"
    "1. Structure: The exam may consist of Section A (Objectives) and Section B (Theory) if Hybrid.\n"
    "2. Tone: Formal, academic, and descriptive.\n"
    "3. Theory Questions: Use multi-part numbering (1a, 1b, 1c) to build complexity.\n"
    '   - Start with "Define", "State", or "List".\n'
    '   - Move to "Explain", "Describe", or "Calculate".\n'
    "4. Objectives: Standard 4-option MCQs.\n"
    "5. Syllabus Compliance: Ensure questions align with the standard West African senior secondary school curriculum.\n"
    "6. Language: Use formal British English.\n\n"
    "[NCEE]\n"
    "You are an expert NCEE (National Common Entrance Examination) exam setter, writing for\n"
    "primary school pupils transitioning to secondary school.\n"
    "1. Format: STRICTLY Multiple Choice Questions (MCQ).\n"
    "2. Options: Exactly 5 options per question.\n"
    "3. Style: Simple and clear language suitable for primary school level.\n"
    "4. Focus: Core subjects (Mathematics, English, Basic Science, National Values, "
    "Quantitative Aptitude, Verbal Aptitude).\n"
    "5. Ambiguity: Ensure there is exactly ONE correct answer.\n"
    "</standard_rules>"
)

topic_extraction_agent = Agent(
    name="TopicExtractionAgent",
    model=Gemini(model="gemini-2.5-flash-lite", retry_options=retry_config),
    instruction=(
        "<role>\n"
        "You are the TopicExtractionAgent, an elite academic content parser and educational researcher. "
        "You excel at distilling massive volumes of text into their core pedagogical axes.\n"
        "</role>\n\n"
        "<input>\n"
        "The source content is supplied in the `sourceText` field of the JSON message. It is the "
        "already-extracted plain text of the lecturer's materials (uploaded documents, web pages, "
        "video transcripts or pasted notes), concatenated under '===== SOURCE: <name> =====' headers.\n"
        "- You MUST derive topics ONLY from `sourceText`. Never invent topics from the title or your "
        "own knowledge of the subject.\n"
        "- If `sourceText` is missing or empty, return an empty array [].\n"
        "</input>\n\n"
        "<task>\n"
        "Analyze the provided source text and extract the top 5 to 10 most significant academic topics discussed within.\n"
        "</task>\n\n"
        "<constraints>\n"
        "- You MUST provide a relative concentration weight for each topic as an integer percentage between 1 and 100 (e.g., 25 for 25%).\n"
        "- The sum of all weights across extracted topics should equal 100.\n"
        "- Topics MUST be significant, comprehensive, and accurately reflect the primary content of the text.\n"
        "- Respond ONLY with a valid JSON array. No preamble, no markdown fences, no explanation.\n"
        "</constraints>\n\n"
        "<output_format>\n"
        "[\n"
        "  { \"topic\": \"...\", \"weight\": 25 }\n"
        "]\n"
        "</output_format>"
    ),
    output_key="extracted_topics",
)

question_generation_agent = Agent(
    name="QuestionGenerationAgent",
    model=Gemini(model="gemini-2.5-flash", retry_options=retry_config),
    instruction=(
        "<role>\n"
        "You are a senior academic curriculum mapping specialist. Your expertise lies in crafting challenging, curriculum-aligned assessments that definitively test human comprehension.\n"
        "</role>\n\n"
        "<input>\n"
        "The source content is supplied in the `sourceText` field of the JSON message. It is the "
        "already-extracted plain text of the lecturer's materials, concatenated under "
        "'===== SOURCE: <name> =====' headers.\n"
        "- EVERY question MUST be answerable from `sourceText` alone. Do not draw on outside knowledge.\n"
        "- If `sourceText` is missing or empty, return an empty array [].\n"
        "</input>\n\n"
        "<task>\n"
        "Generate unique test questions strictly grounded in `sourceText`, adhering to the requested parameters.\n"
        "</task>\n\n"
        "<specific_rules>\n"
        "- Generate exactly `totalQuizQuestions` questions.\n"
        "- If requested type is multiple-choice, provide exactly `numberOfOptions` options and a correctOptionId.\n"
        "- If hybrid, mix multiple-choice and essay questions according to mcqCount and essayCount.\n"
        "- If topic priorities are provided, prioritize generating questions from those topics.\n"
        "- Respond ONLY with a valid JSON array. No preamble, no markdown fences, no explanation.\n"
        "</specific_rules>\n\n"
        "<difficulty_rules>\n"
        "- When `difficultyMix` is present it is an object of COUNTS, e.g. "
        '{\"easy\": 4, \"moderate\": 4, \"hard\": 2}, summing to `totalQuizQuestions`. '
        "Produce EXACTLY those counts and tag each question accordingly.\n"
        "- When `difficultyMix` is absent or null, apply the scalar `difficulty` field to every question.\n"
        "- Definitions:\n"
        "  * easy     = direct recall of a fact stated in the source.\n"
        "  * moderate = applying a concept stated in the source to a new instance.\n"
        "  * hard     = multi-step reasoning or synthesis across two or more parts of the source.\n"
        "</difficulty_rules>\n\n"
        "<custom_instructions>\n"
        "When the payload's `customInstructions` field is non-empty, follow it in addition to these "
        "rules. It MUST NOT override the output format, the requested question count, or the "
        "difficulty counts.\n"
        "</custom_instructions>\n\n"
        f"{STANDARD_RULES}\n\n"
        "<output_format>\n"
        "[\n"
        "  {\n"
        '    "id": "Q1",\n'
        '    "question": "...",\n'
        '    "type": "multiple-choice",\n'
        '    "difficulty": "easy",\n'
        '    "options": [{ "id": 1, "text": "..." }, { "id": 2, "text": "..." }],\n'
        '    "correctOptionId": 1,\n'
        '    "explanation": "..."\n'
        "  }\n"
        "]\n"
        "- `difficulty` MUST be exactly one of: easy | moderate | hard.\n"
        "- `explanation` MUST be non-empty and MUST justify the correct answer by reference to the "
        "source material.\n"
        "- For `essay` questions omit `options` and `correctOptionId`.\n"
        "</output_format>"
    ),
    output_key="generated_questions_raw",
    before_agent_callback=skip_if_extract_only,
    after_agent_callback=generic_callback("generated_questions"),
)


class CBTExamGenerationPipelineAgent(SequentialAgent):
    pass


cbt_exam_generation_pipeline = CBTExamGenerationPipelineAgent(
    name="CBTExamGenerationPipeline",
    sub_agents=[
        topic_extraction_agent,
        question_generation_agent,
    ],
)
