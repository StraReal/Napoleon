== **Napoleon** ==

Just a simple strategy game, I made this to hopefully train an AI (Napoleon would be the AI) on it, to see just how good it could get.

For now, for testing (and playing ;) purposes, I added bots, I tried my best but there's much they could improve on and I don't think any player will find themselves in any real challenge against them.

-- **How to Play** --

Each player starts in a corner, and the cell they start in is their **Capital**, which produces twice as much income and, if conquered, will lead to the collapse of the 'country' (the further a region is from the capital's position, the more likely it is to leave the country at each turn).

Each player has an Income, which is simply the sum of the resource values of the provinces occupied by them. To see the resource value of each province you can either click on it and the value will be displayed next to the province's position, or enable Resource Map view, which is much more intuitive (or activate the Resource Map overlay, but I wouldn't recommend it.

To occupy provinces, you can either expand into them from a neighboring province if they're unoccupied or enter them with your troops, and the province's resource output value will be added to your income.

Army also has an upkeep, which by default is 1 per troop, and each round it's subtracted from the income. If a player can't pay their army's upkeep, troops will each have a chance of starving each round.

There's no real "winning" yet, but I suppose total world domination could be it.

Also, to make the world feel more alive and less easy to conquer, at the start of the game the map will be filled with tribes, passive entities which will hold troops in their towns and mountains, to make the map feel like something to be navigated around.

Variables like prices are purely arbitrary and can be changed around however, I *REALLY* hope I didn't hard-code any in.
-- **Controls** --

Left Click - Select province

E - Expand from selected province. This will automatically expand to the highest-valued (in resources) neighboring province or, if the highest ones are tied, it will choose randomly.

R - Train troop in selected province.

T/Right Click - Select/deselect troop. If one is already selected, pressing again while having selected a neighboring province to the selected troop's will create a movement, which, at the start of the next turn, will move the troops and, when needed, make them fight.

B - Disband troop, which will give back half the cost of buying one. Will also cancel any outgoing movements from target province, so can be used strategically.

N - Enable Resource Map view (seeing the Map with colors and numbers depending on the resource output value of a province)

M - Enable Military Map view (seeing the Map with colors and numbers depending on the amount of troops stationed in a province)

K - Enable Resource Map overlay, which will show resource output values above the normal map.

L - Enable Military Map overlay, which will show amounts of stationed troops above the normal map.


If you find *any* bugs I'd really appreciate if you could open an issue or, even better, fork and send a PR.

I'm not thinking of adding the AI until the gameplay is finalized and the bots aren't dumb anymore.
