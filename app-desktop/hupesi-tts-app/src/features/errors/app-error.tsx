import { Button } from "@/components/ui/button";
import {
  ErrorActions,
  ErrorDescription,
  ErrorHeader,
  ErrorView,
} from "@/features/errors/error-base";

export default function AppErrorPage() {
  return (
    <ErrorView>
      <ErrorHeader>The interface encountered an error</ErrorHeader>
      <ErrorDescription>
        Reload to continue. Saved history and voices will not be lost.
      </ErrorDescription>
      <ErrorActions>
        <Button onClick={() => window.location.reload()} size="lg">
          Reload
        </Button>
      </ErrorActions>
    </ErrorView>
  );
}
