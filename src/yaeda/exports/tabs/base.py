from abc import ABC, abstractmethod


class HTMLTab(ABC):
    def __init__(self, id: str, title: str):
        self._id = id
        self._title = title

    @property
    def head(self) -> str:
        if not self.has_report():
            return ""

        return f"""<button id="button-tab-{self._id}" class="tab-btn" onclick="switchTab('tab-{self._id}', event)">
            {self._title}
        </button>"""

    @abstractmethod
    def has_report(self) -> bool:
        pass

    @abstractmethod
    def _generate(self, *args, **kwargs) -> str:
        pass

    def generate(self, *args, **kwargs) -> str:
        if not self.has_report():
            return ""

        content = self._generate(*args, **kwargs)
        return f"""<div id="tab-{self._id}" class="tab-content">{content}</div>"""
