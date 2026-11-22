import pathlib
import subprocess
import sys
import tempfile
import time
import unittest

from scripts.run_benchmark_pilot import select_units, stop_server, wait_child


class ProcessLifecycleTests(unittest.TestCase):
    def test_explicit_selection_runs_factconsolidation_first(self):
        units = [{'name': 'banking77', 'context_characters': 1},
                 {'name': 'factconsolidation', 'context_characters': 100}]
        result = select_units(units, 'factconsolidation,banking77', 'selection')
        self.assertEqual([unit['name'] for unit in result], ['factconsolidation', 'banking77'])
        self.assertEqual(select_units(units)[0]['name'], 'banking77')

    def test_unknown_selection_fails_without_starting_processes(self):
        with self.assertRaisesRegex(ValueError, 'Unknown or empty'):
            select_units([{'name': 'banking77', 'context_characters': 1}], 'typo', 'selection')

    def test_timeout_reaps_owned_child(self):
        process = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'])
        with self.assertRaises(TimeoutError):
            wait_child(process, time.monotonic() + 0.1)
        self.assertIsNotNone(process.poll())

    def test_stop_request_reaps_owned_child(self):
        process = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'])
        with self.assertRaises(InterruptedError):
            wait_child(process, time.monotonic() + 60, lambda: True)
        self.assertIsNotNone(process.poll())

    def test_service_shutdown_waits_for_stop_signal_and_preserves_database(self):
        with tempfile.TemporaryDirectory() as folder:
            directory = pathlib.Path(folder)
            code = ('import pathlib,time,json,sys; p=pathlib.Path(sys.argv[1]); '
                    '(p/"database").mkdir(); '
                    '\nwhile not (p/"stop.request").exists(): time.sleep(.01)'
                    '\n(p/"status.json").write_text(json.dumps({"phase":"stopped","database_preserved":True}))')
            process = subprocess.Popen([sys.executable, '-c', code, folder])
            shutdown = stop_server(process, directory)
            self.assertEqual(shutdown['launcher_exit_code'], 0)
            self.assertTrue(shutdown['service_status']['database_preserved'])
            self.assertTrue((directory / 'database').exists())
