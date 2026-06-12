import hashlib
import os
from pathlib import Path
from uuid import uuid4

import requests


EMBEDDINGS_URL = (
    "https://storage.googleapis.com/"
    "artifacts.gjones-webinar.appspot.com/embeddings.pkl"
)
EMBEDDINGS_SHA256 = (
    "0331e16d863953ab90d26fa3a2a16fe963990553216fd465d5a0d08f4e002c58"
)
EMBEDDINGS_SIZE = 625199795
DOWNLOAD_CHUNK_SIZE = 1024 * 1024
DOWNLOAD_TIMEOUT = (10, 60)


class ArtifactVerificationError(ValueError):
    pass


def verify_artifact(path, expected_sha256, expected_size):
    artifact_path = Path(path)
    actual_size = artifact_path.stat().st_size
    if actual_size != expected_size:
        raise ArtifactVerificationError(
            f"Artifact size mismatch: expected {expected_size}, got {actual_size}."
        )

    digest = hashlib.sha256()
    with artifact_path.open("rb") as artifact_file:
        for chunk in iter(lambda: artifact_file.read(DOWNLOAD_CHUNK_SIZE), b""):
            digest.update(chunk)

    if digest.hexdigest() != expected_sha256:
        raise ArtifactVerificationError(
            "Artifact checksum mismatch; refusing to load untrusted content."
        )
    return artifact_path


def download_verified_artifact(
    url,
    destination,
    expected_sha256,
    expected_size,
    request_get=None,
):
    destination_path = Path(destination)
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = destination_path.with_name(
        f".{destination_path.name}.{uuid4().hex}.tmp"
    )
    response = None
    request_get = request_get or requests.get

    try:
        response = request_get(
            url,
            stream=True,
            timeout=DOWNLOAD_TIMEOUT,
        )
        response.raise_for_status()
        if not response.url.startswith("https://"):
            raise ArtifactVerificationError(
                "Artifact download must remain on HTTPS."
            )

        digest = hashlib.sha256()
        actual_size = 0
        with temporary_path.open("xb") as artifact_file:
            for chunk in response.iter_content(chunk_size=DOWNLOAD_CHUNK_SIZE):
                if not chunk:
                    continue
                actual_size += len(chunk)
                if actual_size > expected_size:
                    raise ArtifactVerificationError(
                        "Artifact download exceeded the expected size."
                    )
                digest.update(chunk)
                artifact_file.write(chunk)

        if actual_size != expected_size:
            raise ArtifactVerificationError(
                f"Artifact size mismatch: expected {expected_size}, got {actual_size}."
            )
        if digest.hexdigest() != expected_sha256:
            raise ArtifactVerificationError(
                "Artifact checksum mismatch; refusing to save untrusted content."
            )

        os.replace(temporary_path, destination_path)
        return destination_path
    finally:
        try:
            if response is not None:
                response.close()
        finally:
            temporary_path.unlink(missing_ok=True)


def ensure_verified_embeddings(path, request_get=None):
    artifact_path = Path(path)
    if artifact_path.exists():
        return verify_artifact(
            artifact_path,
            EMBEDDINGS_SHA256,
            EMBEDDINGS_SIZE,
        )
    return download_verified_artifact(
        EMBEDDINGS_URL,
        artifact_path,
        EMBEDDINGS_SHA256,
        EMBEDDINGS_SIZE,
        request_get=request_get,
    )
