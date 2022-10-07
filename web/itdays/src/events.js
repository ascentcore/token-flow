const listeners = {
  global: [],
};

export function registerListener(event, listener) {
  if (!listeners[event]) {
    listeners[event] = [];
  }
  listeners[event].push(listener);
}

export function unregisterListener(listener) {
  Object.keys(listeners).forEach((event) => {
    const index = listeners[event].indexOf(listener);
    if (index >= 0) {
      listeners[event].splice(index, 1);
    }
  });
}

export function triggerEvent(event, ...args) {
  if (listeners[event]) {
    listeners[event].forEach((listener) => listener(...args));
  }
}
