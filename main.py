#!/usr/bin/env python3
from __future__ import annotations

import os
import re
import sys
import unicodedata
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence

try:
    from dotenv import load_dotenv
except ImportError:
    print("Error: falta la dependencia 'python-dotenv'. Instala con: pip install python-dotenv")
    sys.exit(1)

try:
    from openai import OpenAI
except ImportError:
    print("Error: falta la dependencia 'openai'. Instala con: pip install openai")
    sys.exit(1)

MODEL_NAME = "gpt-4.1-mini"
MAX_GENERATION_ATTEMPTS = 5
MAX_SENTENCE_ATTEMPTS = 12
COHERENCE_CONTEXT_SIZE = 3
SENTENCE_TEMPERATURE = 0.2
REPAIR_TEMPERATURE = 0.0

SPANISH_ALPHABET = [
    "A",
    "B",
    "C",
    "D",
    "E",
    "F",
    "G",
    "H",
    "I",
    "J",
    "K",
    "L",
    "M",
    "N",
    "Ñ",
    "O",
    "P",
    "Q",
    "R",
    "S",
    "T",
    "U",
    "V",
    "W",
    "X",
    "Y",
    "Z",
]

LETTER_TO_NUMBER = {letter: i + 1 for i, letter in enumerate(SPANISH_ALPHABET)}
NUMBER_TO_LETTER = {i + 1: letter for i, letter in enumerate(SPANISH_ALPHABET)}

WORD_PATTERN = re.compile(r"[A-Za-zÁÉÍÓÚÜÑáéíóúüñ]+(?:[-'][A-Za-zÁÉÍÓÚÜÑáéíóúüñ]+)*")
SENTENCE_SPLIT_PATTERN = re.compile(r"(?<=[.!?])\s+|[\r\n]+")


@dataclass
class ValidationResult:
    is_valid: bool
    sentences: list[str]
    counts: list[int]
    errors: list[str]


@dataclass
class GenerationResult:
    text: str | None
    validation: ValidationResult | None
    attempts_used: int
    errors: list[str]


@dataclass
class ExtractionResult:
    sentences: list[str]
    counts: list[int]
    letters: list[str]
    reconstructed_message: str
    invalid_positions: list[int]


def load_configuration() -> str:
    env_path = Path(".env")
    if not env_path.exists():
        raise FileNotFoundError("No se encontró el archivo .env en el directorio actual.")

    load_dotenv(dotenv_path=env_path, override=False)
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise EnvironmentError("La variable OPENAI_API_KEY no está definida o está vacía en .env.")

    return api_key


def create_openai_client() -> OpenAI:
    api_key = load_configuration()
    return OpenAI(api_key=api_key)


def normalize_text_base(text: str) -> str:
    # Eliminamos marcas diacríticas conservando la Ñ.
    protected = text.replace("Ñ", "__ENYE_UPPER__").replace("ñ", "__ENYE_LOWER__")
    normalized = unicodedata.normalize("NFD", protected)
    no_marks = "".join(ch for ch in normalized if unicodedata.category(ch) != "Mn")
    return no_marks.replace("__ENYE_UPPER__", "Ñ").replace("__ENYE_LOWER__", "ñ")


def normalize_message_for_encoding(message: str) -> tuple[str, list[str]]:
    uppercase = normalize_text_base(message.upper())
    normalized_chars: list[str] = []
    unsupported_chars: list[str] = []

    for char in uppercase:
        if char in LETTER_TO_NUMBER:
            normalized_chars.append(char)
        elif char.isspace():
            # Los espacios no se codifican.
            continue
        else:
            unsupported_chars.append(char)

    return "".join(normalized_chars), unsupported_chars


def letters_to_numbers(message: str) -> list[int]:
    numbers: list[int] = []
    for char in message:
        if char not in LETTER_TO_NUMBER:
            raise ValueError(f"Carácter no soportado durante la codificación: {char!r}")
        numbers.append(LETTER_TO_NUMBER[char])
    return numbers


def numbers_to_letters(numbers: Sequence[int], invalid_placeholder: str = "?") -> tuple[list[str], list[int]]:
    letters: list[str] = []
    invalid_positions: list[int] = []

    for idx, value in enumerate(numbers, start=1):
        letter = NUMBER_TO_LETTER.get(value)
        if letter is None:
            letters.append(invalid_placeholder)
            invalid_positions.append(idx)
        else:
            letters.append(letter)

    return letters, invalid_positions


def split_sentences(text: str) -> list[str]:
    cleaned = text.strip()
    if not cleaned:
        return []

    cleaned = re.sub(r"[ \t]+", " ", cleaned)
    sentences = [part.strip() for part in SENTENCE_SPLIT_PATTERN.split(cleaned) if part.strip()]

    if len(sentences) <= 1:
        sentences = [part.strip() for part in re.split(r"[;:]+", cleaned) if part.strip()]

    return sentences


def count_words(sentence: str) -> int:
    return len(WORD_PATTERN.findall(sentence))


def validate_text_sequence(text: str, expected_sequence: Sequence[int]) -> ValidationResult:
    sentences = split_sentences(text)
    counts = [count_words(sentence) for sentence in sentences]
    errors: list[str] = []

    if len(sentences) != len(expected_sequence):
        errors.append(
            f"Cantidad de frases incorrecta: esperado {len(expected_sequence)}, obtenido {len(sentences)}."
        )

    common_len = min(len(sentences), len(expected_sequence))
    for i in range(common_len):
        expected = expected_sequence[i]
        obtained = counts[i]
        if obtained != expected:
            errors.append(f"Frase {i + 1}: esperado {expected} palabras, obtenido {obtained}.")

    if len(sentences) > len(expected_sequence):
        for i in range(len(expected_sequence), len(sentences)):
            errors.append(f"Frase extra {i + 1}: {counts[i]} palabras.")

    if len(sentences) < len(expected_sequence):
        for i in range(len(sentences), len(expected_sequence)):
            errors.append(f"Falta la frase {i + 1}: se esperaban {expected_sequence[i]} palabras.")

    return ValidationResult(is_valid=not errors, sentences=sentences, counts=counts, errors=errors)


def build_generation_prompt(topic: str, sequence: Sequence[int], attempt: int, previous_errors: Sequence[str]) -> str:
    sequence_list = ", ".join(str(n) for n in sequence)
    per_sentence_rules = "\n".join(
        f"- Frase {idx}: exactamente {value} palabras." for idx, value in enumerate(sequence, start=1)
    )

    feedback = ""
    if previous_errors:
        errors_block = "\n".join(f"- {err}" for err in previous_errors[:8])
        feedback = (
            "\nDetalles de autocorrección (fallos previos detectados por validador externo):\n"
            f"{errors_block}\n"
            "Corrige estos problemas y verifica internamente el conteo antes de responder.\n"
        )

    return f"""1. Objetivo de la tarea
Genera un texto en español, natural y coherente, sobre este tema: "{topic}".
El texto debe codificar una secuencia exacta de longitudes de frase medidas en número de palabras.

2. Rol asignado al modelo
Actúa como redactor humano experto en texto expositivo narrativo en español neutro.

3. Detalles estrictos
- Debes escribir exactamente {len(sequence)} frases.
- Cada frase debe respetar exactamente el número de palabras indicado.
- Mantén coherencia temática entre todas las frases.
- No menciones esteganografía, claves, códigos, instrucciones, validaciones ni reglas.
- No uses listas, numeraciones, títulos ni metacomentarios.
- Evita texto fuera del contenido final.
- Secuencia objetivo: {sequence_list}

Requisitos por frase:
{per_sentence_rules}
{feedback}
4. Formato de salida
- Devuelve un único bloque de texto.
- Solo el texto final.
- Sin explicaciones adicionales.

5. Instrucción de autocorrección o iteración
- Antes de responder, cuenta las palabras de cada frase y compáralas con la secuencia.
- Si alguna frase no cumple, reescribe y vuelve a contar hasta que todas coincidan.
- Solo entrega la versión final correcta.

Intento actual: {attempt}
"""


def build_single_sentence_prompt(
    topic: str,
    target_words: int,
    sentence_index: int,
    total_sentences: int,
    previous_sentences: Sequence[str],
    attempt: int,
    previous_error: str | None,
) -> str:
    context_block = " ".join(previous_sentences).strip()
    context_text = context_block if context_block else "(sin contexto previo)"

    correction_block = ""
    if previous_error:
        correction_block = f"- Error detectado en intento previo: {previous_error}\n"

    special_rule = ""
    if target_words == 1:
        special_rule = (
            "- Caso especial: al ser 1 palabra, la frase puede ser una sola palabra temática seguida de punto.\n"
        )

    return f"""1. Objetivo de la tarea
Escribe UNA sola frase en español, natural y coherente, sobre el tema "{topic}".
Esta frase es la número {sentence_index} de {total_sentences} dentro de un texto mayor.

2. Rol asignado al modelo
Actúa como redactor humano cuidadoso con el conteo exacto de palabras.

3. Detalles estrictos
- Debes escribir exactamente {target_words} palabras.
- Debe ser exactamente una frase.
- Mantén coherencia con el contexto previo.
- No uses listas, numeraciones, títulos ni explicaciones.
- No menciones esteganografía, códigos ni instrucciones.
- Evita dos puntos y punto y coma.
- Usa palabras separadas por espacios simples (sin guiones).
- Termina con un único punto final.
- Contexto previo: {context_text}
{special_rule}{correction_block}
4. Formato de salida
- Devuelve solo la frase final.
- Sin comillas.
- Sin texto adicional.

5. Instrucción de autocorrección o iteración
- Cuenta las palabras antes de responder.
- Si no son exactamente {target_words}, reescribe y vuelve a contar.
- Entrega únicamente una frase válida.

Intento actual para esta frase: {attempt}
"""


def build_sentence_repair_prompt(
    topic: str,
    target_words: int,
    sentence_index: int,
    total_sentences: int,
    candidate_sentence: str,
    previous_sentences: Sequence[str],
    attempt: int,
) -> str:
    context_block = " ".join(previous_sentences).strip()
    context_text = context_block if context_block else "(sin contexto previo)"
    return f"""1. Objetivo de la tarea
Reescribe una frase para que tenga exactamente {target_words} palabras.
La frase corresponde a la posición {sentence_index} de {total_sentences} en un texto sobre "{topic}".

2. Rol asignado al modelo
Actúa como corrector de estilo y conteo exacto en español.

3. Detalles estrictos
- Debes conservar el sentido principal de la frase candidata.
- Debe quedar natural y coherente con el contexto previo.
- Debe ser exactamente una frase.
- Debe tener exactamente {target_words} palabras.
- No uses listas, numeraciones, títulos ni explicaciones.
- Evita guiones y evita dos puntos.
- Termina con un único punto final.
- Contexto previo: {context_text}
- Frase candidata: "{candidate_sentence}"

4. Formato de salida
- Devuelve solo la frase corregida.
- Sin texto adicional.

5. Instrucción de autocorrección o iteración
- Cuenta palabras antes de responder.
- Si no son {target_words}, vuelve a ajustar.

Intento de reparación: {attempt}
"""


def choose_best_sentence_candidate(text: str, target_words: int) -> str:
    normalized = re.sub(r"\s+", " ", text).strip()
    if not normalized:
        return ""

    sentences = split_sentences(normalized)
    if not sentences:
        return normalized

    # Si el modelo devuelve más de una frase, elegimos la más cercana al conteo objetivo.
    best = min(sentences, key=lambda s: (abs(count_words(s) - target_words), len(s)))
    return best.strip()


def normalize_candidate_sentence(raw_text: str) -> str:
    text = re.sub(r"\s+", " ", raw_text).strip()
    text = text.strip("`\"' ")
    text = re.sub(r"^\d+\s*[\).\:-]\s*", "", text)
    text = text.replace(";", ",").replace(":", ",")

    # Si el modelo devuelve varias frases, tomamos solo la primera para validar.
    pieces = split_sentences(text)
    if pieces:
        text = pieces[0].strip()

    text = text.strip("`\"' ")
    if text and text[-1] not in ".!?":
        text += "."

    return text


def request_text_from_openai(
    client: OpenAI,
    model: str,
    prompt: str,
    temperature: float,
    max_output_tokens: int,
) -> str:
    response = client.responses.create(
        model=model,
        input=prompt,
        temperature=temperature,
        max_output_tokens=max_output_tokens,
    )
    return extract_response_text(response)


def try_repair_sentence(
    client: OpenAI,
    model: str,
    topic: str,
    target_words: int,
    sentence_index: int,
    total_sentences: int,
    candidate_sentence: str,
    previous_sentences: Sequence[str],
    max_repair_attempts: int = 3,
) -> tuple[str | None, str]:
    last_error = "Sin detalles."
    for attempt in range(1, max_repair_attempts + 1):
        prompt = build_sentence_repair_prompt(
            topic=topic,
            target_words=target_words,
            sentence_index=sentence_index,
            total_sentences=total_sentences,
            candidate_sentence=candidate_sentence,
            previous_sentences=previous_sentences,
            attempt=attempt,
        )
        try:
            raw = request_text_from_openai(
                client=client,
                model=model,
                prompt=prompt,
                temperature=REPAIR_TEMPERATURE,
                max_output_tokens=max(32, target_words * 5),
            )
        except Exception as exc:  # noqa: BLE001
            last_error = f"Error de API en reparación: {exc}"
            continue

        if not raw:
            last_error = "Respuesta vacía en reparación."
            continue

        best = choose_best_sentence_candidate(raw, target_words)
        repaired = normalize_candidate_sentence(best if best else raw)
        words = count_words(repaired)
        if words == target_words:
            return repaired, ""

        last_error = f"Reparación con conteo incorrecto: esperado {target_words}, obtenido {words}."

    return None, last_error


def generate_single_sentence(
    client: OpenAI,
    topic: str,
    target_words: int,
    sentence_index: int,
    total_sentences: int,
    previous_sentences: Sequence[str],
    model: str,
    max_sentence_attempts: int = MAX_SENTENCE_ATTEMPTS,
) -> tuple[str | None, str]:
    last_error = "Sin detalles."

    for attempt in range(1, max_sentence_attempts + 1):
        prompt = build_single_sentence_prompt(
            topic=topic,
            target_words=target_words,
            sentence_index=sentence_index,
            total_sentences=total_sentences,
            previous_sentences=previous_sentences,
            attempt=attempt,
            previous_error=last_error if attempt > 1 else None,
        )

        try:
            generated_text = request_text_from_openai(
                client=client,
                model=model,
                prompt=prompt,
                temperature=SENTENCE_TEMPERATURE,
                max_output_tokens=max(32, target_words * 5),
            )
        except Exception as exc:  # noqa: BLE001
            last_error = f"Error de API: {exc}"
            continue

        if not generated_text:
            last_error = "Respuesta vacía del modelo."
            continue

        best_candidate = choose_best_sentence_candidate(generated_text, target_words)
        candidate = normalize_candidate_sentence(best_candidate if best_candidate else generated_text)
        sentence_parts = split_sentences(candidate)
        if len(sentence_parts) != 1:
            last_error = f"Se esperaba una sola frase y se detectaron {len(sentence_parts)}."
            continue

        words = count_words(sentence_parts[0])
        if words != target_words:
            repaired, repair_error = try_repair_sentence(
                client=client,
                model=model,
                topic=topic,
                target_words=target_words,
                sentence_index=sentence_index,
                total_sentences=total_sentences,
                candidate_sentence=sentence_parts[0],
                previous_sentences=previous_sentences,
            )
            if repaired is not None:
                return repaired, ""
            last_error = (
                f"Conteo incorrecto: esperado {target_words}, obtenido {words}. "
                f"{repair_error}"
            )
            continue

        return sentence_parts[0], ""

    return None, last_error


def extract_response_text(response: Any) -> str:
    output_text = getattr(response, "output_text", None)
    if isinstance(output_text, str) and output_text.strip():
        return output_text.strip()

    output = getattr(response, "output", None)
    if not isinstance(output, list):
        return ""

    def get_field(obj: Any, field: str) -> Any:
        if isinstance(obj, dict):
            return obj.get(field)
        return getattr(obj, field, None)

    assistant_messages: list[Any] = []
    for item in output:
        if get_field(item, "type") == "message" and get_field(item, "role") == "assistant":
            assistant_messages.append(item)

    if not assistant_messages:
        return ""

    final_phase_messages = [m for m in assistant_messages if get_field(m, "phase") == "final_answer"]
    target_message = final_phase_messages[-1] if final_phase_messages else assistant_messages[-1]

    content = get_field(target_message, "content")
    if not isinstance(content, list):
        return ""

    chunks: list[str] = []
    for part in content:
        if get_field(part, "type") != "output_text":
            continue

        maybe_text = get_field(part, "text")
        text_value: str | None = None
        if isinstance(maybe_text, str):
            text_value = maybe_text
        elif isinstance(maybe_text, dict):
            maybe_value = maybe_text.get("value")
            if isinstance(maybe_value, str):
                text_value = maybe_value

        if text_value and text_value.strip():
            chunks.append(text_value.strip())

    return "\n".join(chunks).strip()


def generate_text_with_openai(
    client: OpenAI,
    topic: str,
    sequence: Sequence[int],
    model: str = MODEL_NAME,
    max_attempts: int = MAX_GENERATION_ATTEMPTS,
) -> GenerationResult:
    previous_errors: list[str] = []

    for attempt in range(1, max_attempts + 1):
        attempt_sentences: list[str] = []
        attempt_errors: list[str] = []

        for idx, target in enumerate(sequence, start=1):
            context = attempt_sentences[-COHERENCE_CONTEXT_SIZE:]
            sentence, sentence_error = generate_single_sentence(
                client=client,
                topic=topic,
                target_words=target,
                sentence_index=idx,
                total_sentences=len(sequence),
                previous_sentences=context,
                model=model,
                max_sentence_attempts=MAX_SENTENCE_ATTEMPTS,
            )
            if sentence is None:
                attempt_errors.append(f"Frase {idx}: {sentence_error}")
                break
            attempt_sentences.append(sentence)

        if len(attempt_sentences) != len(sequence):
            previous_errors = [f"Intento {attempt}: {err}" for err in attempt_errors]
            continue

        generated_text = " ".join(attempt_sentences)
        validation = validate_text_sequence(generated_text, sequence)
        if validation.is_valid:
            return GenerationResult(
                text=generated_text,
                validation=validation,
                attempts_used=attempt,
                errors=[],
            )

        previous_errors = [f"Intento {attempt}: {err}" for err in validation.errors]

    return GenerationResult(
        text=None,
        validation=None,
        attempts_used=max_attempts,
        errors=previous_errors,
    )


def extract_hidden_message(text: str) -> ExtractionResult:
    sentences = split_sentences(text)
    counts = [count_words(sentence) for sentence in sentences]
    letters, invalid_positions = numbers_to_letters(counts, invalid_placeholder="?")
    reconstructed = "".join(letters)
    return ExtractionResult(
        sentences=sentences,
        counts=counts,
        letters=letters,
        reconstructed_message=reconstructed,
        invalid_positions=invalid_positions,
    )


def save_result_file(
    topic: str,
    original_message: str,
    normalized_message: str,
    sequence: Sequence[int],
    generated_text: str,
) -> Path:
    default_filename = f"resultado_estego_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    user_input = input(f"Nombre de archivo .txt (Enter para '{default_filename}'): ").strip()
    filename = user_input if user_input else default_filename

    path = Path(filename)
    if path.suffix.lower() != ".txt":
        path = path.with_suffix(".txt")

    content = (
        f"Tema: {topic}\n\n"
        f"Mensaje original: {original_message}\n"
        f"Mensaje normalizado: {normalized_message}\n"
        f"Secuencia numérica: {', '.join(str(n) for n in sequence)}\n\n"
        "Texto portador generado:\n"
        f"{generated_text}\n"
    )
    path.write_text(content, encoding="utf-8")
    return path


def read_multiline_input(prompt: str) -> str:
    print(prompt)
    print("Pega el texto. Escribe FIN para terminar o deja una línea vacía para finalizar.")
    lines: list[str] = []

    while True:
        try:
            line = input()
        except EOFError:
            break
        except KeyboardInterrupt:
            print("\nEntrada interrumpida.")
            return ""

        if line.strip().upper() == "FIN":
            break
        if line.strip() == "" and lines:
            break
        lines.append(line)

    return "\n".join(lines).strip()


def read_text_from_file() -> str:
    try:
        path_raw = input("Ruta del archivo .txt: ").strip()
    except (EOFError, KeyboardInterrupt):
        print("\nOperación cancelada.")
        return ""

    if not path_raw:
        print("Ruta vacía.")
        return ""

    cleaned_path = path_raw.strip().strip('"').strip("'")
    path = Path(cleaned_path)
    if not path.exists() or not path.is_file():
        print("No se encontró el archivo indicado.")
        return ""

    try:
        raw_content = path.read_text(encoding="utf-8").strip()
    except OSError as exc:
        print(f"No se pudo leer el archivo: {exc}")
        return ""

    marker = "Texto portador generado:"
    if marker in raw_content:
        return raw_content.split(marker, 1)[1].strip()

    return raw_content


def ask_yes_no(prompt: str) -> bool:
    while True:
        try:
            answer = input(prompt).strip().lower()
        except EOFError:
            return False
        except KeyboardInterrupt:
            print()
            return False

        if answer in {"s", "si", "sí", "y", "yes"}:
            return True
        if answer in {"n", "no"}:
            return False
        print("Respuesta no válida. Usa 's' o 'n'.")


def unique_preserve_order(chars: Sequence[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for ch in chars:
        if ch not in seen:
            seen.add(ch)
            ordered.append(ch)
    return ordered


def printable_char(ch: str) -> str:
    if ch == "\n":
        return r"\n"
    if ch == "\t":
        return r"\t"
    if ch == "\r":
        return r"\r"
    if ch == " ":
        return "' '"
    return ch


def handle_generate_option() -> None:
    try:
        original_message = input("Mensaje secreto a ocultar: ").strip()
    except (EOFError, KeyboardInterrupt):
        print("\nOperación cancelada.")
        return

    if not original_message:
        print("El mensaje secreto no puede estar vacío.")
        return

    try:
        topic = input("Tema del texto portador: ").strip()
    except (EOFError, KeyboardInterrupt):
        print("\nOperación cancelada.")
        return

    if not topic:
        print("El tema no puede estar vacío.")
        return

    normalized_message, unsupported = normalize_message_for_encoding(original_message)
    if unsupported:
        unique_unsupported = unique_preserve_order(unsupported)
        formatted = ", ".join(printable_char(ch) for ch in unique_unsupported)
        print("Se detectaron caracteres no soportados:", formatted)
        if not ask_yes_no("¿Quieres eliminarlos y continuar? (s/n): "):
            print("Operación cancelada por el usuario.")
            return

    if not normalized_message:
        print("No quedan letras válidas para codificar tras la normalización.")
        return

    sequence = letters_to_numbers(normalized_message)

    try:
        client = create_openai_client()
    except Exception as exc:  # noqa: BLE001
        print(f"Error de configuración: {exc}")
        return

    print(f"\nGenerando texto portador con el modelo '{MODEL_NAME}'...")
    generation = generate_text_with_openai(
        client=client,
        topic=topic,
        sequence=sequence,
        model=MODEL_NAME,
        max_attempts=MAX_GENERATION_ATTEMPTS,
    )

    if generation.text is None:
        print(f"No se pudo generar un texto válido tras {MAX_GENERATION_ATTEMPTS} intentos.")
        if generation.errors:
            print("Detalle del último error:")
            for err in generation.errors:
                print(f"- {err}")
        return

    print("\n=== TEXTO PORTADOR GENERADO ===")
    print(generation.text)
    print("\n=== DATOS DE CODIFICACIÓN ===")
    print(f"Mensaje normalizado: {normalized_message}")
    print(f"Secuencia numérica: {', '.join(str(n) for n in sequence)}")
    print(f"Intentos usados: {generation.attempts_used}")

    if ask_yes_no("\n¿Quieres guardar el resultado en un archivo .txt? (s/n): "):
        try:
            saved_path = save_result_file(
                topic=topic,
                original_message=original_message,
                normalized_message=normalized_message,
                sequence=sequence,
                generated_text=generation.text,
            )
            print(f"Archivo guardado en: {saved_path.resolve()}")
        except OSError as exc:
            print(f"No se pudo guardar el archivo: {exc}")


def handle_extract_option() -> None:
    print("\nSelecciona origen del texto portador:")
    print("1. Pegar texto manualmente")
    print("2. Cargar desde archivo .txt")

    try:
        source = input("Opción (1-2): ").strip()
    except (EOFError, KeyboardInterrupt):
        print("\nOperación cancelada.")
        return

    if source == "2":
        carrier_text = read_text_from_file()
    else:
        carrier_text = read_multiline_input("\nIntroduce el texto portador multilínea.")

    if not carrier_text:
        print("No se recibió texto para extraer.")
        return

    result = extract_hidden_message(carrier_text)
    if not result.sentences:
        print("No se detectaron frases válidas en el texto proporcionado.")
        return

    print("\n=== RESULTADO DE EXTRACCIÓN ===")
    for idx, (sentence, count, letter) in enumerate(
        zip(result.sentences, result.counts, result.letters),
        start=1,
    ):
        if letter == "?":
            print(f"Frase {idx}: {count} palabras -> INVÁLIDA (fuera de rango 1-27).")
        else:
            print(f"Frase {idx}: {count} palabras -> letra '{letter}'.")
        print(f"  Texto: {sentence}")

    print("\nNúmero de palabras por frase:")
    print(", ".join(str(n) for n in result.counts))
    print("\nLetras obtenidas:")
    print(" ".join(result.letters))
    print("\nMensaje reconstruido:")
    print(result.reconstructed_message)

    if result.invalid_positions:
        invalid_str = ", ".join(str(pos) for pos in result.invalid_positions)
        print(f"\nAdvertencia: frases inválidas en posiciones: {invalid_str}.")


def main_menu() -> None:
    while True:
        print("\n=== MENÚ PRINCIPAL ===")
        print("1. Generar texto portador con mensaje oculto")
        print("2. Extraer mensaje oculto de un texto portador")
        print("3. Salir")

        try:
            option = input("Selecciona una opción (1-3): ").strip()
        except EOFError:
            print("\nFin de entrada. Saliendo.")
            break
        except KeyboardInterrupt:
            print("\nInterrupción detectada. Saliendo.")
            break

        if option == "1":
            handle_generate_option()
        elif option == "2":
            handle_extract_option()
        elif option == "3":
            print("Salida completada.")
            break
        else:
            print("Opción inválida. Debes elegir 1, 2 o 3.")


def main() -> None:
    print("Canal esteganográfico por número de palabras por frase")
    main_menu()


if __name__ == "__main__":
    main()
