from typing import List, Optional, Dict, Any, Union, Callable

import os
import tempfile
import logging
import subprocess
from pathlib import Path
import langdetect
from abc import abstractmethod
import re
import pdb



logger = logging.getLogger(__name__)


def clean_wiki_text(text: str) -> str:
    """
    Clean wikipedia text by removing multiple new lines, removing extremely short lines,
    adding paragraph breaks and removing empty paragraphs
    """
    # get rid of multiple new lines
    # while "\n\n" in text:
    #     text = text.replace("\n\n", "\n")

    # remove extremely short lines
    lines = text.split("\n")
    cleaned = []
    for l in lines:
        if len(l) > 30:
            cleaned.append(l)
        elif l[:2] == "==" and l[-2:] == "==":
            cleaned.append(l)
    text = "\n".join(cleaned)

    # add paragraphs (identified by wiki section title which is always in format "==Some Title==")
    text = text.replace("\n==", "\n\n\n==")

    # remove empty paragrahps
    text = re.sub(r"(==.*==\n\n\n)", "", text)

    return text

class BaseConverter():
    """
    Base class for implementing file converts to transform input documents to text format for ingestion in DocumentStore.
    """

    outgoing_edges = 1

    def __init__(
        self,
        remove_numeric_tables: bool = False,
        valid_languages: Optional[List[str]] = None,
    ):
        """
        :param remove_numeric_tables: This option uses heuristics to remove numeric rows from the tables.
                                      The tabular structures in documents might be noise for the reader model if it
                                      does not have table parsing capability for finding answers. However, tables
                                      may also have long strings that could possible candidate for searching answers.
                                      The rows containing strings are thus retained in this option.
        :param valid_languages: validate languages from a list of languages specified in the ISO 639-1
                                (https://en.wikipedia.org/wiki/ISO_639-1) format.
                                This option can be used to add test for encoding errors. If the extracted text is
                                not one of the valid languages, then it might likely be encoding error resulting
                                in garbled text.
        """

        # # save init parameters to enable export of component config as YAML
        # self.set_config(
        #     remove_numeric_tables=remove_numeric_tables, valid_languages=valid_languages
        # )

        self.remove_numeric_tables = remove_numeric_tables
        self.valid_languages = valid_languages

    @abstractmethod
    def convert(
        self,
        file_path: Path,
        meta: Optional[Dict[str, str]],
        remove_numeric_tables: Optional[bool] = None,
        valid_languages: Optional[List[str]] = None,
        encoding: Optional[str] = "UTF-8",
    ) -> Dict[str, Any]:
        """
        Convert a file to a dictionary containing the text and any associated meta data.
        File converters may extract file meta like name or size. In addition to it, user
        supplied meta data like author, url, external IDs can be supplied as a dictionary.
        :param file_path: path of the file to convert
        :param meta: dictionary of meta data key-value pairs to append in the returned document.
        :param remove_numeric_tables: This option uses heuristics to remove numeric rows from the tables.
                                      The tabular structures in documents might be noise for the reader model if it
                                      does not have table parsing capability for finding answers. However, tables
                                      may also have long strings that could possible candidate for searching answers.
                                      The rows containing strings are thus retained in this option.
        :param valid_languages: validate languages from a list of languages specified in the ISO 639-1
                                (https://en.wikipedia.org/wiki/ISO_639-1) format.
                                This option can be used to add test for encoding errors. If the extracted text is
                                not one of the valid languages, then it might likely be encoding error resulting
                                in garbled text.
        :param encoding: Select the file encoding (default is `utf-8`)
        """
        pass

    def validate_language(self, text: str) -> bool:
        """
        Validate if the language of the text is one of valid languages.
        """
        if not self.valid_languages:
            return True

        try:
            lang = langdetect.detect(text)
        except langdetect.lang_detect_exception.LangDetectException:
            lang = None

        if lang in self.valid_languages:
            return True
        else:
            return False

    def run(self, file_paths: Union[Path, List[Path]],  # type: ignore
            meta: Optional[Union[Dict[str, str], List[Dict[str, str]]]] = None,  # type: ignore
            remove_numeric_tables: Optional[bool] = None,  # type: ignore
            valid_languages: Optional[List[str]] = None):  # type: ignore

        if isinstance(file_paths, Path):
            file_paths = [file_paths]

        if meta is None or isinstance(meta, dict):
            meta = [meta] * len(file_paths)  # type: ignore

        documents: list = []
        for file_path, file_meta in zip(file_paths, meta):
            documents.append(
                self.convert(
                    file_path=file_path,
                    meta=file_meta,
                    remove_numeric_tables=remove_numeric_tables,
                    valid_languages=valid_languages,
                )
            )

        result = {"documents": documents}
        return result, "output_1"

class PDFToTextConverter(BaseConverter):
    def __init__(
        self,
        remove_numeric_tables: bool = False,
        valid_languages: Optional[List[str]] = None,
    ):
        """
        :param remove_numeric_tables: This option uses heuristics to remove numeric rows from the tables.
                                      The tabular structures in documents might be noise for the reader model if it
                                      does not have table parsing capability for finding answers. However, tables
                                      may also have long strings that could possible candidate for searching answers.
                                      The rows containing strings are thus retained in this option.
        :param valid_languages: validate languages from a list of languages specified in the ISO 639-1
                                (https://en.wikipedia.org/wiki/ISO_639-1) format.
                                This option can be used to add test for encoding errors. If the extracted text is
                                not one of the valid languages, then it might likely be encoding error resulting
                                in garbled text.
        """
        # # save init parameters to enable export of component config as YAML
        # self.set_config(
        #     remove_numeric_tables=remove_numeric_tables, valid_languages=valid_languages
        # )

        verify_installation = subprocess.run(["pdftotext -v"], shell=True)
        if verify_installation.returncode == 127:
            raise Exception(
                """pdftotext is not installed. It is part of xpdf or poppler-utils software suite.
                
                   Installation on Linux:
                   wget --no-check-certificate https://dl.xpdfreader.com/xpdf-tools-linux-4.03.tar.gz &&
                   tar -xvf xpdf-tools-linux-4.03.tar.gz && sudo cp xpdf-tools-linux-4.03/bin64/pdftotext /usr/local/bin
                   
                   Installation on MacOS:
                   brew install xpdf
                   
                   You can find more details here: https://www.xpdfreader.com
                """
            )

        super().__init__(
            remove_numeric_tables=remove_numeric_tables, valid_languages=valid_languages
        )

    def convert(
        self,
        file_path: Path,
        meta: Optional[Dict[str, str]] = None,
        remove_numeric_tables: Optional[bool] = None,
        valid_languages: Optional[List[str]] = None,
        encoding: Optional[str] = "Latin1",
    ) -> Dict[str, Any]:
        """
        Extract text from a .pdf file using the pdftotext library (https://www.xpdfreader.com/pdftotext-man.html)
        :param file_path: Path to the .pdf file you want to convert
        :param meta: Optional dictionary with metadata that shall be attached to all resulting documents.
                     Can be any custom keys and values.
        :param remove_numeric_tables: This option uses heuristics to remove numeric rows from the tables.
                                      The tabular structures in documents might be noise for the reader model if it
                                      does not have table parsing capability for finding answers. However, tables
                                      may also have long strings that could possible candidate for searching answers.
                                      The rows containing strings are thus retained in this option.
        :param valid_languages: validate languages from a list of languages specified in the ISO 639-1
                                (https://en.wikipedia.org/wiki/ISO_639-1) format.
                                This option can be used to add test for encoding errors. If the extracted text is
                                not one of the valid languages, then it might likely be encoding error resulting
                                in garbled text.
        :param encoding: Encoding that will be passed as -enc parameter to pdftotext. "Latin 1" is the default encoding
                         of pdftotext. While this works well on many PDFs, it might be needed to switch to "UTF-8" or
                         others if your doc contains special characters (e.g. German Umlauts, Cyrillic characters ...).
                         Note: With "UTF-8" we experienced cases, where a simple "fi" gets wrongly parsed as
                         "xef\xac\x81c" (see test cases). That's why we keep "Latin 1" as default here.
                         (See list of available encodings by running `pdftotext -listenc` in the terminal)
        """

        pages = self._read_pdf(file_path, layout=False, encoding=encoding)
        if remove_numeric_tables is None:
            remove_numeric_tables = self.remove_numeric_tables
        if valid_languages is None:
            valid_languages = self.valid_languages

        cleaned_pages = []
        for page in pages:
            # pdftotext tool provides an option to retain the original physical layout of a PDF page. This behaviour
            # can be toggled by using the layout param.
            #  layout=True
            #      + table structures get retained better
            #      - multi-column pages(eg, research papers) gets extracted with text from multiple columns on same line
            #  layout=False
            #      + keeps strings in content stream order, hence multi column layout works well
            #      - cells of tables gets split across line
            #
            #  Here, as a "safe" default, layout is turned off.
            lines = page.splitlines()
            cleaned_lines = []
            for line in lines:
                words = line.split()
                digits = [word for word in words if any(i.isdigit() for i in word)]

                # remove lines having > 40% of words as digits AND not ending with a period(.)
                if remove_numeric_tables:
                    if (
                        words
                        and len(digits) / len(words) > 0.4
                        and not line.strip().endswith(".")
                    ):
                        logger.debug(f"Removing line '{line}' from {file_path}")
                        continue
                cleaned_lines.append(line)

            page = "\n".join(cleaned_lines)
            cleaned_pages.append(page)

        if valid_languages:
            document_text = "".join(cleaned_pages)
            if not self.validate_language(document_text):
                logger.warning(
                    f"The language for {file_path} is not one of {self.valid_languages}. The file may not have "
                    f"been decoded in the correct text format."
                )

        text = "\f".join(cleaned_pages)
        document = {"content": text, "content_type": "text", "meta": meta}
        return document

    def _read_pdf(
        self, file_path: Path, layout: bool, encoding: Optional[str] = "Latin1"
    ) -> List[str]:
        """
        Extract pages from the pdf file at file_path.
        :param file_path: path of the pdf file
        :param layout: whether to retain the original physical layout for a page. If disabled, PDF pages are read in
                       the content stream order.
        """
        if layout:
            command = ["pdftotext", "-enc", encoding, "-layout", str(file_path), "-"]
        else:
            command = ["pdftotext", "-enc", encoding, str(file_path), "-"]
        output = subprocess.run(command, stdout=subprocess.PIPE, shell=False)  # type: ignore
        document = output.stdout.decode(errors="ignore")
        pages = document.split("\f")
        pages = pages[:-1]  # the last page in the split is always empty.
        return pages

def convert_files_to_dicts(
        dir_path: str,
        clean_func: Optional[Callable] = None,
        split_paragraphs: bool = True,
        encoding: Optional[str] = None
) -> List[dict]:
    """
    Convert all files(.txt, .pdf, .docx) in the sub-directories of the given path to Python dicts that can be written to a
    Document Store.
    :param dir_path: path for the documents to be written to the DocumentStore
    :param clean_func: a custom cleaning function that gets applied to each doc (input: str, output:str)
    :param split_paragraphs: split text in paragraphs.
    :param encoding: character encoding to use when converting pdf documents.
    """
    file_paths = [p for p in Path(dir_path).glob("**/*")]
    allowed_suffixes = [".pdf", ".txt", ".docx"]
    suffix2converter: Dict[str, BaseConverter] = {}

    suffix2paths: Dict[str, List[Path]] = {}
    for path in file_paths:
        file_suffix = path.suffix.lower()
        if file_suffix in allowed_suffixes:
            if file_suffix not in suffix2paths:
                suffix2paths[file_suffix] = []
            suffix2paths[file_suffix].append(path)
        elif not path.is_dir():
            logger.warning('Skipped file {0} as type {1} is not supported here. '
                           'See haystack.file_converter for support of more file types'.format(path, file_suffix))

    # No need to initialize converter if file type not present
    for file_suffix in suffix2paths.keys():
        if file_suffix == ".pdf":
            suffix2converter[file_suffix] = PDFToTextConverter()
        if file_suffix == ".txt":
            suffix2converter[file_suffix] = TextConverter()
        if file_suffix == ".docx":
            suffix2converter[file_suffix] = DocxToTextConverter()

    documents = []
    for suffix, paths in suffix2paths.items():
        for path in paths:
            if encoding is None and suffix == '.pdf':
                encoding = "UTF-8"
            logger.info('Converting {}'.format(path))
            document = suffix2converter[suffix].convert(
                    file_path=path,
                    meta=None,
                    encoding=encoding,
            )
            text = document["content"]

            if clean_func:
                text = clean_func(text)

            if split_paragraphs:
                for para in text.split("\n\n"):
                    if not para.strip():  # skip empty paragraphs
                        continue
                    documents.append({"content": para, "meta": {"name": path.name}})
            else:
                documents.append({"content": text, "meta": {"name": path.name}})

    return documents
