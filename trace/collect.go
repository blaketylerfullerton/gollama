package trace

// Collector implements model.Tracer by accumulating events in memory instead of
// writing them to a file, so a UI can watch a pass as it happens.
//
// The events it builds are identical to the ones a Writer would emit — same
// constructors, same copying — which means a live view and a replayed file view
// are looking at exactly the same data. That's the point: the inspector needs
// only one rendering path.
type Collector struct {
	opts Opts
	tr   Trace
	// on, when set, is called for every event as it arrives. Used to push
	// updates at a UI rather than having it poll.
	on func(Event)
}

func NewCollector(h Header, opts Opts) *Collector {
	h.Version = FormatVersion
	return &Collector{opts: opts.withDefaults(), tr: Trace{Header: h}}
}

// OnEvent registers a callback invoked for each event. It runs on the goroutine
// driving the forward pass, so it should hand off rather than block.
func (c *Collector) OnEvent(fn func(Event)) { c.on = fn }

// Trace returns what has been collected so far.
func (c *Collector) Trace() *Trace { return &c.tr }

// Reset clears the events, keeping the header. Used between generation steps so
// each token gets its own trace.
func (c *Collector) Reset() { c.tr.Events = nil }

func (c *Collector) emit(e Event) {
	e.Seq = len(c.tr.Events)
	c.tr.Events = append(c.tr.Events, e)
	if c.on != nil {
		c.on(e)
	}
}

func (c *Collector) Stage(layer int, name string, x [][]float32) {
	if e, ok := stageEvent(c.opts, layer, name, x); ok {
		c.emit(e)
	}
}

func (c *Collector) Attention(layer, head int, weights [][]float32) {
	if e, ok := attentionEvent(c.opts, layer, head, weights); ok {
		c.emit(e)
	}
}

func (c *Collector) Rotary(layer, head int, before, after []float32) {
	c.emit(rotaryEvent(c.opts, layer, head, before, after))
}

func (c *Collector) Note(layer int, format string, args ...any) {
	c.emit(noteEvent(layer, format, args...))
}

func (c *Collector) LogitLens(layer int, logits []float32) {
	c.emit(lensEvent(c.opts, layer, logits))
}

// Snapshot returns a copy of the collected trace, safe to hand to another
// goroutine while collection continues.
//
// The Events slice is copied, but the Event values inside share their slices with
// the originals. That's fine because those were already copies made at emit
// time, and nothing mutates an Event once it's recorded.
func (c *Collector) Snapshot() *Trace {
	events := make([]Event, len(c.tr.Events))
	copy(events, c.tr.Events)
	return &Trace{Header: c.tr.Header, Events: events}
}
