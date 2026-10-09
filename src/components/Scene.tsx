import { useEffect, useRef, useState } from "react";
import {
  RotateCcw,
  Plus,
  Minus,
  Maximize,
  Move,
  AlertTriangle,
} from "lucide-react";
import { NeuralScene, type SceneState } from "../graphics/NeuralScene";
export default function Scene(props: SceneState & { className?: string }) {
  const host = useRef<HTMLDivElement>(null),
    scene = useRef<NeuralScene | null>(null),
    latest = useRef(props);
  latest.current = props;
  const [failed, setFailed] = useState(false),
    [loading, setLoading] = useState(true),
    [attempt, setAttempt] = useState(0);
  useEffect(() => {
    setLoading(true);
    setFailed(false);
    try {
      scene.current = new NeuralScene(host.current!, latest.current, () =>
        setFailed(true),
      );
      setLoading(false);
    } catch (e) {
      console.error("3D initialization failed", e);
      setFailed(true);
      setLoading(false);
    }
    return () => {
      scene.current?.dispose();
      scene.current = null;
    };
  }, [props.mode, attempt]);
  useEffect(() => {
    scene.current?.update(props);
  }, [props]);
  return (
    <div className={`scene ${props.className || ""}`}>
      <div className="scene-host" ref={host} />
      {loading && !failed && (
        <div className="scene-loading" role="status">
          <span className="spinner" />
          Preparing 3D glass…
        </div>
      )}
      {failed && (
        <div className="scene-failure" role="alert">
          <AlertTriangle />
          <strong>3D rendering is unavailable</strong>
          <p>
            The numerical controls still work. Try reloading or a browser with
            WebGL 2 enabled.
          </p>
          <button onClick={() => setAttempt((n) => n + 1)}>Retry 3D</button>
        </div>
      )}
      <div className="scene-foot">
        <span>
          <Move size={13} /> Drag to orbit · pinch to zoom
        </span>
        <div className="camera-tools">
          <button
            aria-label="Zoom in"
            title="Zoom in"
            disabled={failed || loading}
            onClick={() => scene.current?.zoom(0.84)}
          >
            <Plus size={15} />
          </button>
          <button
            aria-label="Zoom out"
            title="Zoom out"
            disabled={failed || loading}
            onClick={() => scene.current?.zoom(1.18)}
          >
            <Minus size={15} />
          </button>
          <button
            aria-label="Fit to screen"
            title="Fit to screen"
            disabled={failed || loading}
            onClick={() => scene.current?.fit()}
          >
            <Maximize size={15} />
          </button>
          <button
            aria-label="Reset camera"
            title="Reset camera"
            disabled={failed || loading}
            onClick={() => scene.current?.fit()}
          >
            <RotateCcw size={15} />
          </button>
        </div>
      </div>
    </div>
  );
}
