import { app } from "../../../scripts/app.js";

app.registerExtension({
    name: "AdvancedRelightNode.UI.FinalComplete",
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name === "RelightNode") {

            const onExecuted = nodeType.prototype.onExecuted;
            nodeType.prototype.onExecuted = function(message) {
                onExecuted?.apply(this, arguments);
                
                if (message?.previews) {
                    const imageUrl = message.previews[0];
                    
                    const img = new Image();
                    img.src = imageUrl + `?t=${new Date().getTime()}`;

                    img.onload = () => {
                        const anchorWidget = this.widgets.find(w => w.name === "ui_anchor");
                        
                        this.previewImage = img;
                        
                        this.applyAnchorResize();
                        const newSize = this.computeSize();
                        this.size[1] = newSize[1];
                        this.setDirtyCanvas(true, true);
                        delete anchorWidget.computeSize;
                        this.drawSlider();
                    };
                    
                    img.onerror = () => {
                        console.error(`[Relight Node] ERREUR: Impossible de charger l'image depuis l'URL directe: ${imageUrl}`);
                    };
                }
            };

            const onNodeCreated = nodeType.prototype.onNodeCreated;
            nodeType.prototype.onNodeCreated = function () {
                onNodeCreated?.apply(this, arguments);

                const node = this;

                this.lights = [{}, {}];
                this.draggedLightIndex = null;
                node._customFlags ??= {};
                if (node._customFlags.hasImageResizeRatio === undefined)
                    node._customFlags.hasImageResizeRatio = false;
                node._onetimeFlags ??= {};
                if (node._onetimeFlags.haspassed === undefined)
                    node._onetimeFlags.haspassed = false;
                    node._onetimeFlags.counter = 0;
                this.previewImage = null;
                this.imageRect = { x: 0, y: 0, w: 0, h: 0 }; 

                requestAnimationFrame(() => {
                    if (this.sliderInitialized) return;
                    this.sliderInitialized = true;
                    this.lights[0].xWidget = this.widgets.find((w) => w.name === "light_x_1");
                    this.lights[0].yWidget = this.widgets.find((w) => w.name === "light_y_1");
                    this.lights[0].typeWidget = this.widgets.find(w => w.name === "light_type_1");
                    this.lights[0].pointSizeWidget = this.widgets.find(w => w.name === "point_size_1");
                    this.lights[0].intensityWidget = this.widgets.find(w => w.name === "intensity_1");
                    this.lights[0].neonWidgets = this.widgets.filter(w => ["neon_length_1", "neon_angle_1"].includes(w.name));
                    const enableLight2Widget = this.widgets.find(w => w.name === "enable_light_2");
                    this.lights[1].xWidget = this.widgets.find((w) => w.name === "light_x_2");
                    this.lights[1].yWidget = this.widgets.find((w) => w.name === "light_y_2");
					this.lights[1].zWidget = this.widgets.find(w => w.name === "light_z_2");
                    this.lights[1].typeWidget = this.widgets.find(w => w.name === "light_type_2");
                    this.lights[1].pointSizeWidget = this.widgets.find(w => w.name === "point_size_2");
                    this.lights[1].intensityWidget = this.widgets.find(w => w.name === "intensity_2");
                    this.lights[1].neonWidgets = this.widgets.filter(w => ["neon_length_2", "neon_angle_2"].includes(w.name));
                    const anchorWidget = this.widgets.find((w) => w.name === "ui_anchor");
                    const maskEnabledWidget = this.widgets.find(w => w.name === "mask_enabled");
                    const maskWidgets = this.widgets.filter(w => ["brush_intensity", "brush_softness", "mask_intensity_mult", "mask_gamma_mult"].includes(w.name));
                    const spacer1 = this.widgets.find((w) => w.name === "spacer_1");
					const spacer2 = this.widgets.find((w) => w.name === "spacer_2");
                    const lightX1 = this.widgets.find(w => w.name === "light_x_1");
                    const lightY1 = this.widgets.find(w => w.name === "light_y_1");
					const lightX2 = this.widgets.find(w => w.name === "light_x_2");
                    const lightY2 = this.widgets.find(w => w.name === "light_y_2");
					
                    if (!anchorWidget || !anchorWidget.inputEl) { return; }

                    lightX1.hidden = true;
                    lightX1.computeSize = () => [0, -4]; // Supprime l'espace vertical
                    lightY1.hidden = true;
                    lightY1.computeSize = () => [0, -4]; // Supprime l'espace vertical
					lightX2.hidden = true;
                    lightX2.computeSize = () => [0, -4]; // Supprime l'espace vertical
                    lightY2.hidden = true;
                    lightY2.computeSize = () => [0, -4]; // Supprime l'espace vertical
                    spacer1.hidden = true;
					spacer2.hidden = true;
					
                    anchorWidget.computeSize = (width) => [width, 256];
                    
					const sliderCanvas = document.createElement("canvas");
                    sliderCanvas.style.width = "100%";
                    sliderCanvas.style.height = "100%";
                    sliderCanvas.style.cursor = "pointer";
                    const container = anchorWidget.inputEl.parentElement;
                    container.replaceChild(sliderCanvas, anchorWidget.inputEl);
                    container.style.padding = "0";
                    const ctx = sliderCanvas.getContext("2d");

                    const drawSlider = () => {
                        const rect = sliderCanvas.getBoundingClientRect();
                        if (sliderCanvas.width !== rect.width || sliderCanvas.height !== rect.height) {
                            sliderCanvas.width = rect.width;
                            sliderCanvas.height = rect.height;
                        }
                        const width = sliderCanvas.width;
                        const height = sliderCanvas.height;
                        ctx.fillStyle = "rgba(0,0,0,1)";
                        ctx.fillRect(0, 0, width, height);
                        if (node.previewImage && node.previewImage.complete) {
                            const imgRatio = node.previewImage.naturalWidth / node.previewImage.naturalHeight;
                            const canvasRatio = width / height;
                            let destWidth, destHeight, destX, destY;
                            if (imgRatio > canvasRatio) {
                                destWidth = width; destHeight = width / imgRatio;
                                destX = 0; destY = (height - destHeight) / 2;
                            } else {
                                destHeight = height; destWidth = height * imgRatio;
                                destY = 0; destX = (width - destWidth) / 2;
                            }
                            ctx.drawImage(node.previewImage, destX, destY, destWidth, destHeight);
                            node.imageRect = { x: destX, y: destY, w: destWidth, h: destHeight };
                        } else {
                            node.imageRect = { x: 0, y: 0, w: width, h: height };
                        }
                        node.lights.forEach((light, index) => {
                            if (index > 0 && !enableLight2Widget.value) return;
                            const lightCenterX = node.imageRect.x + (light.xWidget.value * node.imageRect.w);
                            const lightCenterY = node.imageRect.y + (light.yWidget.value * node.imageRect.h);
                            ctx.save();
                            ctx.globalCompositeOperation = 'overlay';
                            if (light.typeWidget.value === "Point") {
								let radius;

                                if (index === 0) { 
                                   radius = (light.pointSizeWidget.value * 100) + (light.intensityWidget.value * 3);
                                } else { 
                                   radius = (light.pointSizeWidget.value * 500) + (light.intensityWidget.value * 3);
                                }
								
                                const gradient = ctx.createRadialGradient(lightCenterX, lightCenterY, 0, lightCenterX, lightCenterY, radius);
                                gradient.addColorStop(0, "rgba(255, 255, 240, 0.9)");
                                gradient.addColorStop(1, "rgba(255, 255, 240, 0)");
                                ctx.fillStyle = gradient;
                                ctx.beginPath();
                                ctx.arc(lightCenterX, lightCenterY, radius, 0, 2 * Math.PI);
                                ctx.fill();
                            }
                            ctx.restore();
                            Object.assign(ctx, { fillStyle: "rgba(255,255,255,0.8)", strokeStyle: "black", lineWidth: 2 });
                            ctx.beginPath();
                            ctx.arc(lightCenterX, lightCenterY, 7, 0, 2 * Math.PI);
                            ctx.fill();
                            ctx.stroke();
                        });
                    };
                    this.drawSlider = drawSlider;

                    const applyAnchorResize = () => {
                        node.canvasHeight = sliderCanvas.getBoundingClientRect().height;
                        const nodeContentWidth = node.size[0] - 20;
                        if (node.previewImage) {
                            node.aspectRatio = node.previewImage.naturalHeight / node.previewImage.naturalWidth;
                            const newHeight = nodeContentWidth * node.aspectRatio
                            anchorWidget.computeSize = (w) => [w, newHeight];
                        } else {
                            //if (!node._onetimeFlags.haspassed && (node._onetimeFlags.counter === 1 || node._onetimeFlags.counter === 2)) {
                                node._onetimeFlags.goodratio = 256 / node.canvasHeight;
                                node._onetimeFlags.haspassed = true;
                            //}
                            node._onetimeFlags.counter++;
                            const ratio = node._onetimeFlags.goodratio;
                            anchorWidget.computeSize = (width) => [width, node.canvasHeight * ratio];
                        }
                    };
                    this.applyAnchorResize = applyAnchorResize;
                    
                    const onDragMove = (event) => {
                        if (!isDragging || node.draggedLightIndex === null) return;
                        const rect = sliderCanvas.getBoundingClientRect();
                        const mouseX = event.clientX - rect.left;
                        const mouseY = event.clientY - rect.top;
                        let valueX = (mouseX - node.imageRect.x) / node.imageRect.w;
                        let valueY = (mouseY - node.imageRect.y) / node.imageRect.h;
                        const activeLight = node.lights[node.draggedLightIndex];
                        activeLight.xWidget.value = Math.max(0, Math.min(1, valueX));
                        activeLight.yWidget.value = Math.max(0, Math.min(1, valueY));
                        drawSlider();
                    };
                    let isDragging = false;
                    const onDragEnd = () => {
                        if (!isDragging) return;
                        isDragging = false;
                        if (node.draggedLightIndex !== null) {
                            const activeLight = node.lights[node.draggedLightIndex];
                            activeLight.xWidget.setValue(activeLight.xWidget.value);
                            activeLight.yWidget.setValue(activeLight.yWidget.value);
                            node.draggedLightIndex = null;
                        }
                        window.removeEventListener("mousemove", onDragMove);
                        window.removeEventListener("mouseup", onDragEnd);
                        node.setDirtyCanvas(true);
                    };
                    sliderCanvas.addEventListener("mousedown", (event) => {
                        const rect = sliderCanvas.getBoundingClientRect();
                        const mouseX = event.clientX - rect.left;
                        const mouseY = event.clientY - rect.top;
                        let closestIndex = 0;
                        let minDistance = Infinity;
                        node.lights.forEach((light, index) => {
                            if (index > 0 && !enableLight2Widget.value) return;
                            const lightX = node.imageRect.x + light.xWidget.value * node.imageRect.w;
                            const lightY = node.imageRect.y + light.yWidget.value * node.imageRect.h;
                            const distance = Math.sqrt(Math.pow(mouseX - lightX, 2) + Math.pow(mouseY - lightY, 2));
                            if (distance < minDistance) {
                                minDistance = distance;
                                closestIndex = index;
                            }
                        });
                        node.draggedLightIndex = closestIndex;
                        isDragging = true;
                        window.addEventListener("mousemove", onDragMove);
                        window.addEventListener("mouseup", onDragEnd);
                        event.preventDefault();
                        event.stopPropagation();
                        onDragMove(event);
                    });
                    new ResizeObserver(drawSlider).observe(container);
                    const setWidgetVisibility = (widget, isHidden) => {
                        if (!widget) return;
                        widget.hidden = isHidden;
                        if (isHidden) {
                            widget.computeSize = () => [0, -4];
                        } else {
                            delete widget.computeSize;
                        }
                    };
                    const toggleLightWidgets = (light, shouldHide) => {
                        if (!light.typeWidget) return;
                        const isNeon = light.typeWidget.value === "Neon";
                        light.neonWidgets.forEach(w => setWidgetVisibility(w, shouldHide || !isNeon));
                        setWidgetVisibility(light.pointSizeWidget, shouldHide || isNeon);
                        //setWidgetVisibility(light.xWidget, shouldHide);
                        //setWidgetVisibility(light.yWidget, shouldHide);
						setWidgetVisibility(light.zWidget, shouldHide);
                        setWidgetVisibility(light.typeWidget, shouldHide);
                        setWidgetVisibility(light.intensityWidget, shouldHide);
                    };
                    const toggleMaskWidgets = () => {
                        const isEnabled = maskEnabledWidget.value;
                        maskWidgets.forEach(w => setWidgetVisibility(w, !isEnabled));
                        applyAnchorResize();
                        const newSize = node.computeSize();
                        node.size[1] = newSize[1];
                        node.setDirtyCanvas(true, true);
                        delete anchorWidget.computeSize;
                    };
                    const updateAllWidgetsVisibility = () => {
                        toggleLightWidgets(node.lights[0], false);
                        toggleLightWidgets(node.lights[1], !enableLight2Widget.value);
                        applyAnchorResize();
                        const newSize = node.computeSize();
                        node.size[1] = newSize[1];
                        node.setDirtyCanvas(true, true);
                        delete anchorWidget.computeSize;
                    };
                    enableLight2Widget.callback = () => { updateAllWidgetsVisibility(); drawSlider(); };
                    if(maskEnabledWidget) maskEnabledWidget.callback = toggleMaskWidgets;
                    this.lights[0].typeWidget.callback = updateAllWidgetsVisibility;
                    this.lights[1].typeWidget.callback = updateAllWidgetsVisibility;
                    setTimeout(() => {
                        updateAllWidgetsVisibility();
                        toggleMaskWidgets();
                        drawSlider();
                    }, 50);
                });
            };
        }
    },
});