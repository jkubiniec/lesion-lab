"""Execute report JavaScript against a small DOM contract, without a browser.

This verifies data binding and event behavior. It does not verify visual layout,
browser compatibility, actual downloads, or assistive-technology behavior.
"""
from html.parser import HTMLParser
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]


def report_fixture(*, control):
    """Exercise report binding without depending on retained research run folders."""
    assay = {"samples": 2, "trials": 4, "exact_copy_successes": 8 if control else 0,
             "second_generation_successes": 2 if control else 0,
             "qualifying_samples": 2 if control else 0,
             "per_sample": [{"slot": 0, "source_hex": "2b2d3c3e" * 16,
                             "exact_copy_successes": 4 if control else 0}]}
    conditions = {name: {"post": {"overall_accuracy": .8 - index * .1},
                         "lesioned_units": list(range(index))}
                  for index, name in enumerate(("sham", "executive", "random"))}
    doses = [{"lesion_count": count,
              "executive_lesion": {"summary": {"overall_accuracy": {"mean": 1 - count * .1}}},
              "random_nonexecutive_lesion": {"summary": {"overall_accuracy": {"mean": .9 - count * .1}},
                  "all_motor_units_removed_masks": count // 4, "n_masks": 8}}
             for count in range(5)]
    return {"manifest": {"mode": "engineered_positive_control" if control else "random_origin",
        "population": 16, "warmup_interactions": 100, "seed": 42, "elapsed_seconds": .1,
        "copy_gate": {"passed": control, "candidate_slots": [0, 1] if control else []}},
        "baseline": assay,
        "arms": {name: {"immediate": assay, "final": assay} for name in
                 (("sham", "focal", "random") if control else ())},
        "warmup": [{"interaction": n, "unique_tapes": 16 - n // 20,
            "byte_entropy": 4 + n / 100, "exact_copy_events": n if control else 0,
            "active_ops": n + 2, "executed_steps": n * 32} for n in (0, 50, 100)],
        "executive": {"conditions": conditions, "dose_response": doses} if control else None}


class ReportParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.elements = []
        self.scripts = []
        self.in_script = False
        self.in_metric = False
        self.option = None
        self.options = []

    def handle_starttag(self, tag, attributes):
        attrs = dict(attributes)
        if attrs.get("id"):
            self.elements.append({"tag": tag, "attributes": attrs})
        if tag == "script":
            self.in_script = True
            self.scripts.append("")
        if tag == "select" and attrs.get("id") == "metric":
            self.in_metric = True
        if tag == "option" and self.in_metric:
            self.option = {"value": attrs["value"], "text": ""}
            self.options.append(self.option)

    def handle_endtag(self, tag):
        if tag == "script":
            self.in_script = False
        if tag == "select":
            self.in_metric = False
        if tag == "option":
            self.option = None

    def handle_data(self, data):
        if self.in_script:
            self.scripts[-1] += data
        if self.option is not None:
            self.option["text"] += data


NODE_HARNESS = r"""
const vm = require('node:vm');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const input = JSON.parse(fs.readFileSync(0, 'utf8'));
const all = [], ids = new Map();
class Element {
  constructor(tag) {
    this.tagName = tag.toLowerCase(); this.children = []; this.attributes = {};
    this.listeners = {}; this._text = ''; this.className = ''; this.hidden = false;
    this.clientWidth = 610; this.style = {setProperty(k,v) {this[k] = String(v);}};
    this.classList = {add: name => {this.className += ' ' + name;}};
    this._value = ''; this.metricOptions = []; this.selectedOptions = [];
    all.push(this);
  }
  set id(value) {this._id = String(value); ids.set(this._id, this);}
  get id() {return this._id || '';}
  set value(value) {
    this._value = value;
    this.selectedOptions = this.metricOptions.filter(option => option.value === value);
  }
  get value() {return this._value;}
  set textContent(value) {this._text = String(value); this.children = [];}
  get textContent() {return this._text + this.children.map(c => c.textContent).join('');}
  setAttribute(key, value) {
    this.attributes[key] = String(value);
    if (key === 'id') this.id = value;
    if (key === 'class') this.className = value;
    if (key === 'hidden') this.hidden = true;
    if (key === 'tabindex') this.tabIndex = Number(value);
  }
  getAttribute(key) {return this.attributes[key] ?? null;}
  append(...children) {this.children.push(...children);}
  replaceChildren(...children) {this.children = [...children]; this._text = '';}
  addEventListener(type, callback) {(this.listeners[type] ||= []).push(callback);}
  dispatch(type, event = {}) {
    for (const callback of this.listeners[type] || []) callback(event);
  }
  click() {this.dispatch('click');}
  focus() {document.activeElement = this;}
}
const document = {
  getElementById(id) {assert(ids.has(id), `missing DOM id ${id}`); return ids.get(id);},
  createElement(tag) {return new Element(tag);},
  createElementNS(namespace, tag) {return new Element(tag);},
  querySelectorAll(selector) {
    assert.equal(selector, '[role=tab]', `unsupported test DOM selector ${selector}`);
    return all.filter(node => node.attributes.role === 'tab');
  },
  activeElement: null
};
for (const spec of input.elements) {
  const e = new Element(spec.tag);
  for (const [key,value] of Object.entries(spec.attributes)) e.setAttribute(key, value ?? '');
}
const metric = document.getElementById('metric');
metric.metricOptions = input.options;
metric.value = input.options[0].value;
const createdURLs = [], revokedURLs = [];
const context = vm.createContext({document,
  Blob: class Blob {constructor(parts, options) {this.parts = parts; this.options = options;}},
  URL: {createObjectURL(blob) {assert.equal(blob.options.type, 'application/json');
                              createdURLs.push(blob); return 'blob:test-report';},
        revokeObjectURL(url) {revokedURLs.push(url);}},
  setTimeout(callback) {callback(); return 1;}
});
const script = new vm.Script(input.script, {filename: input.name});
script.runInContext(context, {timeout: 10000});
const data = vm.runInContext('D', context);
const node = id => document.getElementById(id);
const descendants = (parent, tag) => parent.children.flatMap(child =>
  [...(child.tagName === tag ? [child] : []), ...descendants(child, tag)]);
const percent = value => (100 * Number(value || 0)).toFixed(1) + '%';
const numericText = value => Number(value || 0).toLocaleString(undefined, {maximumFractionDigits: 2});
function checkFinite() {
  for (const e of all) {
    for (const [key,value] of Object.entries(e.attributes))
      assert(!/NaN|Infinity|undefined/.test(value), `${e.tagName}.${key}=${value}`);
    for (const [key,value] of Object.entries(e.style))
      if (typeof value !== 'function') assert(!/NaN|Infinity|undefined/.test(String(value)), `${key}=${value}`);
  }
}
assert.equal(node('stats').children.length, 4);
assert.equal(node('stats').children[0].children[0].textContent, numericText(data.manifest.population));
assert.equal(node('stats').children[1].children[0].textContent,
             numericText(data.manifest.checkpoint_interaction ?? data.manifest.warmup_interactions));
const expectedAssayRows = (data.baseline ? 1 : 0) + 2 * Object.keys(data.arms).length;
assert.equal(node('assay-table').children.length, expectedAssayRows);
if (data.baseline) {
  const row = node('assay-table').children[0];
  assert.equal(row.children[2].textContent,
               `${numericText(data.baseline.exact_copy_successes)} / ${numericText(data.baseline.samples * data.baseline.trials)}`);
}
if (data.manifest.mode === 'engineered_positive_control') {
  assert.equal(node('mode').textContent, 'ENGINEERED COPYING CONTROL');
  assert(node('status').textContent.includes(data.manifest.copy_gate.passed ? 'not spontaneous emergence' : 'deferred'));
} else {
  assert(node('status').textContent.includes(data.manifest.copy_gate.passed ? 'Copying verified' : 'not verified'));
}
const tabs = document.querySelectorAll('[role=tab]');
const checkTab = selected => {
  assert.equal(tabs.filter(tab => tab.getAttribute('aria-selected') === 'true').length, 1);
  for (const tab of tabs) {
    const isSelected = tab === selected;
    assert.equal(tab.getAttribute('aria-selected'), String(isSelected));
    assert.equal(node(tab.getAttribute('aria-controls')).hidden, !isSelected);
  }
};
checkTab(tabs[0]); tabs[1].click(); checkTab(tabs[1]);
let prevented = false;
tabs[1].dispatch('keydown', {key: 'ArrowRight', preventDefault() {prevented = true;}});
assert(prevented); checkTab(tabs[2]); assert.equal(document.activeElement, tabs[2]);
tabs[2].dispatch('keydown', {key: 'ArrowRight', preventDefault() {}}); checkTab(tabs[0]);
tabs[0].dispatch('keydown', {key: 'ArrowLeft', preventDefault() {}}); checkTab(tabs[2]);
for (const option of input.options) {
  metric.value = option.value; metric.dispatch('change');
  const points = data.warmup.filter(row => Number.isFinite(row[option.value]));
  if (points.length) assert(node('chart-description').textContent.startsWith(option.text + ':'));
  checkFinite();
}
const circle = all.find(e => e.tagName === 'circle' && e.listeners.pointerenter);
if (circle) {circle.dispatch('pointerenter'); circle.dispatch('pointerleave'); checkFinite();}
let conditionRows = 0, doseRows = 0;
if (data.executive) {
  const tables = descendants(node('control-content'), 'table');
  conditionRows = Object.keys(data.executive.conditions || {}).length;
  assert.equal(tables.length, data.executive.dose_response ? 2 : 1);
  const actualRows = tables[0].children.find(child => child.tagName === 'tbody').children;
  assert.equal(actualRows.length, conditionRows);
  for (const [index,[condition,record]] of Object.entries(Object.entries(data.executive.conditions || {}))) {
    const row = actualRows[Number(index)];
    assert.equal(row.children[0].textContent, condition);
    assert.equal(row.children[1].textContent, percent(record.post?.overall_accuracy));
  }
  if (data.executive.dose_response) {
    const actualDoses = tables[1].children.find(child => child.tagName === 'tbody').children;
    doseRows = data.executive.dose_response.length;
    assert.equal(actualDoses.length, doseRows);
    for (let i = 0; i < doseRows; ++i) {
      const expected = data.executive.dose_response[i], actual = actualDoses[i].children;
      assert.equal(actual[0].textContent, String(expected.lesion_count));
      assert.equal(actual[1].textContent, percent(expected.executive_lesion.summary.overall_accuracy.mean));
      assert.equal(actual[2].textContent, percent(expected.random_nonexecutive_lesion.summary.overall_accuracy.mean));
      assert.equal(actual[3].textContent,
                   `${expected.random_nonexecutive_lesion.all_motor_units_removed_masks} / ${expected.random_nonexecutive_lesion.n_masks}`);
    }
  }
} else assert(node('control-content').textContent.includes('not included'));
node('download').click(); assert.equal(createdURLs.length, 1); assert.equal(revokedURLs.length, 1);
assert.equal(JSON.parse(createdURLs[0].parts[0]).manifest.seed, data.manifest.seed);
checkFinite();
console.log(JSON.stringify({assayRows: expectedAssayRows, conditionRows, doseRows,
                           tabs: tabs.length, metricOptions: input.options.length}));
"""


class ReportBehaviorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.node = shutil.which("node")
        if not cls.node:
            raise unittest.SkipTest("Node.js is required for report JavaScript checks")

    def evaluate(self, path):
        parser = ReportParser()
        parser.feed(path.read_text())
        self.assertEqual(len(parser.scripts), 1)
        self.assertNotIn("__LAB_DATA__", parser.scripts[0], "generated report data is missing")
        result = subprocess.run([self.node, "-e", NODE_HARNESS], input=json.dumps({
            "name": str(path), "elements": parser.elements, "options": parser.options,
            "script": parser.scripts[0]}), text=True, capture_output=True, timeout=20)
        self.assertEqual(result.returncode, 0, result.stderr)
        return json.loads(result.stdout)

    def evaluate_fixture(self, *, control):
        template = (ROOT / "report.html").read_text()
        with tempfile.TemporaryDirectory(prefix="report-fixture-") as folder:
            path = Path(folder) / "report.html"
            path.write_text(template.replace("__LAB_DATA__", json.dumps(report_fixture(control=control))))
            return self.evaluate(path)

    def test_generated_control_fixture_data_and_interactions(self):
        summary = self.evaluate_fixture(control=True)
        self.assertEqual(summary, {"assayRows": 7, "conditionRows": 3, "doseRows": 5,
                                   "tabs": 3, "metricOptions": 5})

    def test_random_origin_fixture_handles_deferred_injury(self):
        summary = self.evaluate_fixture(control=False)
        self.assertEqual(summary["assayRows"], 1)

    def test_template_javascript_parses(self):
        parser = ReportParser()
        parser.feed((ROOT / "report.html").read_text())
        self.assertEqual(len(parser.scripts), 1)
        # Compiling a Script validates JavaScript syntax without evaluating the
        # template's data placeholder or inventing browser behavior.
        code = "const fs=require('node:fs'),vm=require('node:vm');new vm.Script(fs.readFileSync(0,'utf8'));"
        result = subprocess.run([self.node, "-e", code], input=parser.scripts[0],
                                text=True, capture_output=True, timeout=10)
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
