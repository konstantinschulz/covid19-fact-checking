from typing import Any
from config import Config
import time
import docker
from docker.models.containers import Container
from elg.model.response.ClassificationResponse import ClassificationResponse
from elg.service import Service
from elg.model import TextRequest
import unittest

from elg_service import CovidCredibilityClassifier


class ElgTestCase(unittest.TestCase):
    content: str = "".join([
        "Im Rahmen der Corona-Krise verlangt das linke Spektrum einen Lastenausgleich. ",
        "Bedeutet: Entschädigungslose Enteignung durch den Staat. Die Presse verbreitet die Idee mit positiven Ton. ",
        "Auch ein Ökonom findet das gut."])
    # TODO: check why the scores slightly differ in local and docker setup
    score: float = 0.04978

    def test_local(self):
        request: TextRequest = TextRequest(content=ElgTestCase.content)
        service: CovidCredibilityClassifier = CovidCredibilityClassifier(Config.COVID_CREDIBILITY_CLASSIFIER)
        cr: ClassificationResponse = service.process_text(request)
        self.assertEqual(cr.classes[-1].score, ElgTestCase.score)
        self.assertEqual(type(cr), ClassificationResponse)

    def test_docker(self):
        client = docker.from_env()
        ports_dict: dict = dict()
        ports_dict[Config.DOCKER_PORT_CREDIBILITY] = Config.HOST_PORT_CREDIBILITY
        container: Container = client.containers.run(
            Config.DOCKER_IMAGE_CREDIBILITY_SERVICE, ports=ports_dict, detach=True, environment=dict(TRANSFORMERS_OFFLINE=1))
        # wait for the container to start the API
        time.sleep(1)
        service: Service = Service.from_local_installation(
            Config.DOCKER_COMPOSE_SERVICE_NAME, f"http://localhost:{Config.HOST_PORT_CREDIBILITY}")
        response: Any = service(ElgTestCase.content, sync_mode=True)
        cr: ClassificationResponse = response
        container.stop()
        container.remove()
        self.assertEqual(cr.classes[-1].score, ElgTestCase.score)
        self.assertEqual(type(response), ClassificationResponse)

    def test_elg_remote(self):
        service: Service = Service.from_id(18690, auth_file="token.json")  # scope="offline_access", use_cache=False
        response: Any = service(ElgTestCase.content)
        cr: ClassificationResponse = response
        self.assertEqual(cr.classes[-1].score, ElgTestCase.score)
        self.assertEqual(type(response), ClassificationResponse)


if __name__ == '__main__':
    unittest.main()
