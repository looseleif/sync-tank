(() => {
  "use strict";
  const department = document.querySelector("#department");
  const products = [...document.querySelectorAll(".product")];
  const count = document.querySelector("#catalog-count");
  function filter() {
    let visible = 0;
    for (const product of products) {
      product.hidden = department.value !== "all" && product.dataset.department !== department.value;
      if (!product.hidden) visible++;
    }
    count.textContent = `${visible} ${visible === 1 ? "experiment" : "experiments"} on the shelf`;
  }
  department.addEventListener("change", filter);
  document.querySelector(".catalog-toolbar").hidden = false;
  filter();
})();
