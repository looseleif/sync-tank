"""Browser regression checks; run against a local or deployed static site."""
import argparse
from io import BytesIO
import json
from pathlib import Path

from PIL import Image
from playwright.sync_api import sync_playwright

parser = argparse.ArgumentParser()
parser.add_argument('base_url')
parser.add_argument('--channel', default=None)
args = parser.parse_args()
base = args.base_url.rstrip('/')
root = Path(__file__).resolve().parents[2]
shots = root.parent / 'builder-checks'
shots.mkdir(exist_ok=True)
key = 'sync-tank-builder-v1'

with sync_playwright() as p:
    browser = p.chromium.launch(channel=args.channel)
    context = browser.new_context(viewport={'width': 1440, 'height': 1000}, accept_downloads=True)
    page = context.new_page()
    errors, failed = [], []
    page.on('pageerror', lambda error: errors.append(str(error)))
    page.on('response', lambda response: failed.append(response.url) if response.status >= 400 else None)
    page.goto(base + '/builder.html')
    page.wait_for_function('!document.getElementById("add").disabled', timeout=20000)

    def state():
        return json.loads(page.evaluate('(key) => localStorage.getItem(key)', key))

    def range_value(name, value):
        page.locator('#' + name).focus()
        page.locator('#' + name).evaluate('(el, value) => { el.value = value; el.dispatchEvent(new Event("input", {bubbles:true})); el.dispatchEvent(new Event("change", {bubbles:true})); }', str(value))

    def stage_click(x=0.5, y=0.5):
        page.locator('#stage canvas').click(position={'x': page.locator('#stage').bounding_box()['width'] * x, 'y': page.locator('#stage').bounding_box()['height'] * y})

    assert state()['items'] == []
    empty = page.locator('#stage canvas').screenshot()
    assert len(Image.open(BytesIO(empty)).convert('RGB').getcolors(1000000)) > 100, 'Blank 3D canvas'
    page.select_option('#view', 'front')
    page.click('#add')
    assert not state()['items'][0]['placed']
    stage_click(0.45, 0.55)
    assert state()['items'][0]['placed']
    range_value('x', 0.4)
    range_value('y', 0.65)
    assert abs(state()['items'][0]['position']['x'] - 0.4) < 0.001
    assert abs(state()['items'][0]['position']['y'] - 0.65) < 0.001
    page.select_option('#view', 'top')
    stage_click(0.6, 0.45)
    assert abs(state()['items'][0]['position']['y'] - 0.65) < 0.001, 'Plane move reset height'
    before_deselect = state()
    page.locator('#stage canvas').dblclick(position={'x': 30, 'y': 30})
    assert not page.locator('#inspector').is_visible()
    assert state() == before_deselect, 'Double-click deselect changed geometry'
    page.select_option('#kind', 'floater')
    page.click('#add')
    page.select_option('#mount', 'x+')
    range_value('y', 0.7)
    assert state()['items'][1]['position']['x'] == 1
    page.select_option('#mount', 'z-')
    range_value('x', 0.45)
    floater = state()['items'][1]
    assert floater['position']['z'] == 0 and abs(floater['position']['y'] - 0.7) < 0.001
    page.select_option('#kind', 'endo')
    page.click('#add')
    range_value('x', 0.55)
    range_value('yaw', 35)
    range_value('pitch', -15)
    assert state()['items'][2]['yaw'] == 35
    page.click('#undo')
    assert state()['items'][2]['pitch'] == 0
    page.click('#redo')
    assert state()['items'][2]['pitch'] == -15
    print('PASS unplaced objects, placement planes, synchronized sliders, mounts, aiming, undo/redo', flush=True)

    page.locator('#items button').first.click()
    old_z = state()['items'][0]['position']['z']
    page.set_input_files('#photo-file', root / 'images/sync.jpg')
    page.wait_for_selector('#reference', state='visible')
    page.select_option('#photo-mode', 'crop')
    box = page.locator('#reference').bounding_box()
    page.mouse.move(box['x'] + box['width'] * .1, box['y'] + box['height'] * .2)
    page.mouse.down()
    page.mouse.move(box['x'] + box['width'] * .9, box['y'] + box['height'] * .8, steps=10)
    page.mouse.up()
    crop = state()['references']['front']['crop']
    assert abs(crop[0] - .1) < .02 and abs(crop[1] - .1) < .02, crop
    page.select_option('#photo-mode', 'place')
    page.locator('#reference').click(position={'x': box['width'] * .4, 'y': box['height'] * .5})
    placed = state()['items'][0]['position']
    assert abs(placed['x'] - .375) < .03 and abs(placed['y'] - .5) < .03, placed
    assert placed['z'] == old_z, 'Front photo changed unobserved depth'
    page.fill('#notes', 'Estimated from front photo; depth needs side view.')
    page.locator('#notes').blur()
    with page.expect_download() as info:
        page.click('#export')
    exported = json.loads(Path(info.value.path()).read_text())
    assert exported['items'][0]['notes'].startswith('Estimated')
    assert 'data:image' not in json.dumps(exported) and 'blob:' not in json.dumps(exported)
    assert exported['references']['front']['crop'] == crop
    before_bad = state()
    bad = dict(exported, tank=dict(exported['tank'], width=-2))
    page.set_input_files('#import', {'name': 'invalid.json', 'mimeType': 'application/json', 'buffer': json.dumps(bad).encode()})
    page.wait_for_function('document.getElementById("save-status").textContent.startsWith("Import rejected")')
    assert state() == before_bad
    page.reload()
    page.wait_for_function('!document.getElementById("add").disabled')
    assert state()['items'] == before_bad['items']
    assert 'reload photo' in page.locator('#photo-name').inner_text()
    assert not page.locator('#photo-tools').is_visible()
    page.set_input_files('#import', {'name': 'valid.json', 'mimeType': 'application/json', 'buffer': json.dumps(exported).encode()})
    page.wait_for_function('document.getElementById("save-status").textContent.startsWith("Draft saved")')
    assert state()['items'] == exported['items']
    print('PASS local photo crop/placement, preserved depth, export, rejected invalid import, restore without image bytes', flush=True)

    page.select_option('#view', 'orbit')
    page.select_option('#interaction', 'orbit')
    before = page.locator('#stage canvas').screenshot()
    box = page.locator('#stage canvas').bounding_box()
    page.mouse.move(box['x'] + box['width'] * .65, box['y'] + box['height'] * .3)
    page.mouse.down()
    page.mouse.move(box['x'] + box['width'] * .8, box['y'] + box['height'] * .45, steps=12)
    page.mouse.up()
    assert before != page.locator('#stage canvas').screenshot(), 'Orbit did not change canvas'
    for width, height in [(1440, 1000), (1024, 768), (390, 844), (320, 740)]:
        page.set_viewport_size({'width': width, 'height': height})
        page.select_option('#view', 'orbit')
        page.locator('#items button').last.click()
        page.evaluate('scrollTo(0, 0)')
        assert page.evaluate('document.documentElement.scrollWidth <= innerWidth'), f'Overflow {width}'
        if width > 600:
            assert page.evaluate('document.documentElement.scrollHeight <= innerHeight'), 'Desktop editor does not fit the screen'
        canvas = page.locator('#stage canvas').screenshot()
        assert len(Image.open(BytesIO(canvas)).convert('RGB').getcolors(1000000)) > 100
        page.screenshot(path=str(shots / f'builder-{width}.png'), full_page=True)
        print(f'PASS 3D canvas and controls at {width}x{height}', flush=True)

    page.goto(base + '/hardware.html')
    page.locator('.bench-photo img').scroll_into_view_if_needed()
    page.locator('.bench-photo img').evaluate('(image) => image.decode()')
    assert page.locator('.connection-diagram').count() == 5
    for heading in ['Camera data takes the local route', 'Analog video needs a capture adapter',
                    'IP cameras take a different path', 'PoE powers a compatible receiver',
                    'The servo has signal and power']:
        assert page.get_by_role('heading', name=heading, exact=True).count() == 1
    for width in [1440, 390, 320]:
        page.set_viewport_size({'width': width, 'height': 900})
        assert page.evaluate('document.documentElement.scrollWidth <= innerWidth')
    page.set_viewport_size({'width': 1440, 'height': 1000})
    page.screenshot(path=str(shots / 'hardware.png'), full_page=True)
    assert not errors, errors
    assert not failed, failed
    print('PASS hardware diagrams, image, responsive layout, no script or HTTP errors', flush=True)
    isolated = browser.new_context()
    unavailable = isolated.new_page()
    unavailable.route('**/three.module.js', lambda route: route.abort())
    unavailable.goto(base + '/builder.html')
    unavailable.wait_for_function('document.getElementById("save-status").textContent.startsWith("Editor unavailable")')
    assert unavailable.locator('#add').is_disabled()
    damaged = browser.new_context()
    damaged.add_init_script('localStorage.setItem("sync-tank-builder-v1", "broken draft")')
    recovered = damaged.new_page()
    recovered.goto(base + '/builder.html')
    recovered.wait_for_function('!document.getElementById("add").disabled')
    recovered.click('#add')
    assert recovered.evaluate('localStorage.getItem("sync-tank-builder-v1")') == 'broken draft'
    assert 'not been overwritten' in recovered.locator('#save-status').inner_text()
    print('PASS missing-3D fallback and preservation of unreadable stored draft', flush=True)
    browser.close()
